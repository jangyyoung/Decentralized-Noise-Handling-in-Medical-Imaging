import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
from torch.autograd import Function
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

##############################################
#  LayerNorm2d 정의: (N, C, H, W) -> (N, H, W, C) 후 LayerNorm 적용
##############################################
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_channels, eps=eps)
    
    def forward(self, x):
        # x: (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        # 다시 (N, C, H, W)로 변환
        x = x.permute(0, 3, 1, 2)
        return x

##############################################
#  wavemix 관련 내부 함수 및 클래스 시작
##############################################

def roll(x, shift, dim):
    return torch.roll(x, shifts=shift, dims=dim)

def sfb1d(lo, hi, g0, g1, mode='zero', dim=-1):
    C = lo.shape[1]
    d = dim % 4
    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(np.copy(np.array(g0).ravel()), dtype=torch.float, device=lo.device)
    if not isinstance(g1, torch.Tensor):
        g1 = torch.tensor(np.copy(np.array(g1).ravel()), dtype=torch.float, device=lo.device)
    L = g0.numel()
    shape = [1,1,1,1]
    shape[d] = L
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)
    s = (2, 1) if d == 2 else (1,2)
    g0 = torch.cat([g0]*C, dim=0)
    g1 = torch.cat([g1]*C, dim=0)
    if mode in ['per', 'periodization']:
        y = F.conv_transpose2d(lo, g0, stride=s, groups=C) + F.conv_transpose2d(hi, g1, stride=s, groups=C)
        if d == 2:
            y[:,:,:L-2] = y[:,:,:L-2] + y[:,:,lo.shape[2]:lo.shape[2]+L-2]
            y = y[:,:,:lo.shape[2]]
        else:
            y[:,:,:,:L-2] = y[:,:,:,:L-2] + y[:,:,:,lo.shape[3]:lo.shape[3]+L-2]
            y = y[:,:,:,:lo.shape[3]]
        y = roll(y, 1 - L//2, dim=dim)
    else:
        pad = (L-2, 0) if d==2 else (0, L-2)
        y = F.conv_transpose2d(lo, g0, stride=s, padding=pad, groups=C) + \
            F.conv_transpose2d(hi, g1, stride=s, padding=pad, groups=C)
    return y

def afb1d(x, h0, h1, mode='zero', dim=-1):
    C = x.shape[1]
    d = dim % 4
    s = (2,1) if d==2 else (1,2)
    N = x.shape[d]
    if not isinstance(h0, torch.Tensor):
        h0 = torch.tensor(np.copy(np.array(h0).ravel()[::-1]), dtype=torch.float, device=x.device)
    if not isinstance(h1, torch.Tensor):
        h1 = torch.tensor(np.copy(np.array(h1).ravel()[::-1]), dtype=torch.float, device=x.device)
    L = h0.numel()
    L2 = L // 2
    shape = [1,1,1,1]
    shape[d] = L
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)
    if mode in ['per', 'periodization']:
        if x.shape[dim] % 2 == 1:
            if d == 2:
                x = torch.cat((x, x[:,:,-1:]), dim=2)
            else:
                x = torch.cat((x, x[:,:,:,-1:]), dim=3)
            N += 1
        x = roll(x, -L2, dim=d)
        pad = (L-1, 0) if d==2 else (0, L-1)
        lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
        N2 = N // 2
        if d == 2:
            lohi[:,:,:L2] = lohi[:,:,:L2] + lohi[:,:,N2:N2+L2]
            lohi = lohi[:,:,:N2]
        else:
            lohi[:,:,:,:L2] = lohi[:,:,:,:L2] + lohi[:,:,:,N2:N2+L2]
            lohi = lohi[:,:,:,:N2]
    else:
        pad = (0,0)
        lohi = F.conv2d(x, h, padding=pad, stride=s, groups=C)
    return lohi

class AFB2D(Function):
    @staticmethod
    def forward(ctx, x, h0_row, h1_row, h0_col, h1_col, mode):
        ctx.save_for_backward(h0_row, h1_row, h0_col, h1_col)
        ctx.shape = x.shape[-2:]
        mode = int_to_mode(mode)
        ctx.mode = mode
        lohi = afb1d(x, h0_row, h1_row, mode=mode, dim=3)
        y = afb1d(lohi, h0_col, h1_col, mode=mode, dim=2)
        s = y.shape
        y = y.reshape(s[0], -1, 4, s[-2], s[-1])
        low = y[:,:,0].contiguous()
        highs = y[:,:,1:].contiguous()
        return low, highs

    @staticmethod
    def backward(ctx, low, highs):
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            h0_row, h1_row, h0_col, h1_col = ctx.saved_tensors
            lh, hl, hh = torch.unbind(highs, dim=2)
            lo = sfb1d(low, lh, h0_col, h1_col, mode=mode, dim=2)
            hi = sfb1d(hl, hh, h0_col, h1_col, mode=mode, dim=2)
            dx = sfb1d(lo, hi, h0_row, h1_row, mode=mode, dim=3)
            if dx.shape[-2] > ctx.shape[-2] and dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:,:,:ctx.shape[-2], :ctx.shape[-1]]
            elif dx.shape[-2] > ctx.shape[-2]:
                dx = dx[:,:,:ctx.shape[-2]]
            elif dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:,:,:,:ctx.shape[-1]]
        return dx, None, None, None, None, None

def mode_to_int(mode):
    if mode=='zero':
        return 0
    elif mode=='symmetric':
        return 1
    elif mode in ['per', 'periodization']:
        return 2
    elif mode=='constant':
        return 3
    elif mode=='reflect':
        return 4
    elif mode=='replicate':
        return 5
    elif mode=='periodic':
        return 6
    else:
        raise ValueError("Unknown pad type: {}".format(mode))

def int_to_mode(mode):
    if mode==0:
        return 'zero'
    elif mode==1:
        return 'symmetric'
    elif mode==2:
        return 'periodization'
    elif mode==3:
        return 'constant'
    elif mode==4:
        return 'reflect'
    elif mode==5:
        return 'replicate'
    elif mode==6:
        return 'periodic'
    else:
        raise ValueError("Unknown pad type: {}".format(mode))

def prep_filt_afb1d(h0, h1, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    h0 = np.array(h0[::-1]).ravel()
    h1 = np.array(h1[::-1]).ravel()
    t = torch.get_default_dtype()
    h0 = torch.tensor(h0, device=device, dtype=t).reshape((1,1,-1))
    h1 = torch.tensor(h1, device=device, dtype=t).reshape((1,1,-1))
    return h0, h1

def prep_filt_afb2d(h0_col, h1_col, h0_row=None, h1_row=None, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    h0_col, h1_col = prep_filt_afb1d(h0_col, h1_col, device)
    if h0_row is None:
        h0_row, h1_row = h0_col, h1_col
    else:
        h0_row, h1_row = prep_filt_afb1d(h0_row, h1_row, device)
    h0_col = h0_col.reshape((1,1,-1,1))
    h1_col = h1_col.reshape((1,1,-1,1))
    h0_row = h0_row.reshape((1,1,1,-1))
    h1_row = h1_row.reshape((1,1,1,-1))
    return h0_col, h1_col, h0_row, h1_row

class DWTForward(nn.Module):
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave)==2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave)==4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]
        filts = prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_col', filts[0])
        self.register_buffer('h1_col', filts[1])
        self.register_buffer('h0_row', filts[2])
        self.register_buffer('h1_row', filts[3])
        self.J = J
        self.mode = mode
    def forward(self, x):
        yh = []
        ll = x
        mode = mode_to_int(self.mode)
        for j in range(self.J):
            ll, high = AFB2D.apply(ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row, mode)
            yh.append(high)
        return ll, yh

# Global DWTForward instances for different levels
device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
xf1 = DWTForward(J=1, mode='zero', wave='db1').to(device_)

class Level1Waveblock(nn.Module):
    def __init__(self, *, mult=2, ff_channel=16, final_dim=16, dropout=0.5):
        super().__init__()
        # Feedforward branch
        self.feedforward = nn.Sequential(
            nn.Conv2d(final_dim, final_dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(final_dim * mult, ff_channel, 1),
            nn.ConvTranspose2d(ff_channel, final_dim, 4, stride=2, padding=1),
            LayerNorm2d(final_dim)
        )
        # Reduction branch
        self.reduction = nn.Conv2d(final_dim, int(final_dim / 4), 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        # Apply reduction
        x_reduced = self.reduction(x)
        # Apply 2D wavelet transform using global xf1
        Y1, Yh = xf1(x_reduced)
        # Reshape the high-frequency component
        x_h = torch.reshape(Yh[0], (b, int(c * 3 / 4), int(h / 2), int(w / 2)))
        # Concatenate low-frequency and high-frequency components
        x_cat = torch.cat((Y1, x_h), dim=1)
        # Process via feedforward branch
        out = self.feedforward(x_cat)
        return out

##############################################
#  wavemix 관련 내부 함수 및 클래스 끝
##############################################

# 1. MaskPredictor: 입력 이미지에서 손상된 영역(마스크)를 예측하는 모듈
class MaskPredictor(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32):
        super(MaskPredictor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(mid_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()  # 마스크 값은 0~1 사이로 제한
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        mask = self.sigmoid(self.conv3(x))
        return mask

# 2. WaveMixModule: 기존 모듈 (mask 입력은 그대로 사용)
class WaveMixModule(nn.Module):
    def __init__(self, *, depth, mult=2, ff_channel=16, final_dim=16, dropout=0.):
        super().__init__()
        # 원래 이미지 채널(3)을 입력으로 받도록 구성
        self.conv = nn.Sequential(
            nn.Conv2d(3, int(final_dim/2), 3, 1, 1),
            nn.Conv2d(int(final_dim/2), final_dim, 3, 2, 1),
        )
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # Level1Waveblock은 내부에 정의된 것으로 사용
            self.layers.append(Level1Waveblock(mult=mult, ff_channel=ff_channel, final_dim=final_dim, dropout=dropout))
        self.depthconv = nn.Sequential(
            nn.Conv2d(final_dim, final_dim, 5, groups=final_dim, padding="same"),
            nn.GELU(),
            LayerNorm2d(final_dim)   # BatchNorm2d -> LayerNorm2d
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(final_dim*2, int(final_dim/2), 4, stride=2, padding=1),
            LayerNorm2d(int(final_dim/2))   # BatchNorm2d -> LayerNorm2d
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(int(final_dim/2) + 3, 3, 1),
        )
    def forward(self, img, mask):
        # x = torch.cat([img, mask], dim=1)  # 사용되지 않음
        x = self.conv(img)
        skip1 = x
        for attn in self.layers:
            x = attn(x) + x
        x = self.depthconv(x)
        x = torch.cat([x, skip1], dim=1)  # skip connection
        x = self.decoder1(x)
        x = torch.cat([x, img], dim=1)      # skip connection
        x = self.decoder2(x)
        return x

# 3. SelfInpaint: 단일 모델에서 손상된 이미지만 입력받아 내부에서 마스크를 예측하고 복원 수행
class SelfInpaint(nn.Module):
    def __init__(self, *, num_modules=1, blocks_per_module=4, mult=4, ff_channel=16, final_dim=16, dropout=0.,
                 mask_predictor_mid_channels=32):
        super().__init__()
        # 내부에서 마스크를 예측할 모듈
        self.mask_predictor = MaskPredictor(in_channels=3, mid_channels=mask_predictor_mid_channels)
        # 여러 WaveMixModule을 쌓아 inpainting 네트워크 구성 (WavePaint의 역할)
        self.wavemodules = nn.ModuleList([])
        for _ in range(num_modules):
            self.wavemodules.append(
                WaveMixModule(
                    depth=blocks_per_module,
                    mult=mult,
                    ff_channel=ff_channel,
                    final_dim=final_dim,
                    dropout=dropout
                )
            )
    def forward(self, img):
        # Step 1: 입력 이미지로부터 손상 영역(마스크) 예측
        predicted_mask = self.mask_predictor(img)
        # predicted_mask는 0~1 사이 값; 1에 가까울수록 손상 영역으로 간주
        x = img
        # 각 WaveMixModule에 보완된 마스크 (1 - predicted_mask)를 전달
        for module in self.wavemodules:
            x = module(x, 1 - predicted_mask) + x
        # 최종 출력: 예측한 마스크 영역은 모델 복원 결과, 나머지 영역은 원본 이미지 유지
        inpainted = x * predicted_mask + img
        return inpainted
