import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
from torch.autograd import Function
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

##############################################
#  LayerNorm2d: (N, C, H, W) -> (N, H, W, C) 후 LayerNorm 적용
##############################################
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_channels, eps=eps)
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

##############################################
#  wavemix 관련 내부 함수 및 클래스 (일부 수정)
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
        mode = mode_to_int(mode)
        ctx.mode = mode
        lohi = afb1d(x, h0_row, h1_row, mode=mode, dim=3)
        y = afb1d(lohi, h0_col, h1_col, mode=mode, dim=2)
        s = y.shape
        y = y.reshape(s[0], -1, 4, s[-2], s[-1])
        LL = y[:,:,0].contiguous()
        LH = y[:,:,1].contiguous()
        HL = y[:,:,2].contiguous()
        HH = y[:,:,3].contiguous()
        return LL, LH, HL, HH

    @staticmethod
    def backward(ctx, LL, LH, HL, HH):
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            h0_row, h1_row, h0_col, h1_col = ctx.saved_tensors
            # 스택으로 고주파 sub-band를 복원에 사용
            highs = torch.stack([LH, HL, HH], dim=2)  # (B, C, 3, H, W)
            lo = sfb1d(LL, highs[:, :, 0], h0_col, h1_col, mode=mode, dim=2)
            hi = sfb1d(highs[:, :, 1], highs[:, :, 2], h0_col, h1_col, mode=mode, dim=2)
            dx = sfb1d(lo, hi, h0_row, h1_row, mode=mode, dim=3)
            if dx.shape[-2] > ctx.shape[-2] and dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:,:,:ctx.shape[-2], :ctx.shape[-1]]
            elif dx.shape[-2] > ctx.shape[-2]:
                dx = dx[:,:,:ctx.shape[-2]]
            elif dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:,:,:,:ctx.shape[-1]]
        return dx, None, None, None, None, None

def mode_to_int(mode):
    if isinstance(mode, int):
        if 0 <= mode <= 6:
            return mode
        else:
            raise ValueError("Unknown pad type: {}".format(mode))
    if mode == 'zero':
        return 0
    elif mode == 'symmetric':
        return 1
    elif mode in ['per', 'periodization']:
        return 2
    elif mode == 'constant':
        return 3
    elif mode == 'reflect':
        return 4
    elif mode == 'replicate':
        return 5
    elif mode == 'periodic':
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
        # J=1인 경우, 네 개의 sub-band로 분리된 결과를 반환합니다.
        LL, LH, HL, HH = AFB2D.apply(x, self.h0_col, self.h1_col, self.h0_row, self.h1_row, self.mode)
        return LL, LH, HL, HH

# Global DWTForward instance
device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
xf1 = DWTForward(J=1, mode='zero', wave='db1').to(device_)

##############################################
#  Swin Transformer Block (간단한 예시)
##############################################
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=3, window_size=7, mlp_ratio=4.0, dropout=0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        #self.attn.in_proj_weight = nn.Parameter(self.attn.in_proj_weight.contiguous())
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        self.window_size = window_size
        self.dim = dim

    def forward(self, x):
        # x: (B, C, H, W) with H, W divisible by window_size
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.window_size, self.window_size, W // self.window_size, self.window_size).contiguous()
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, num_win_H, num_win_W, window_size, window_size, C)
        num_windows = (H // self.window_size) * (W // self.window_size)
        x = x.view(B * num_windows, self.window_size * self.window_size, C).contiguous()  # (B*num_windows, window_area, C)

        shortcut = x
        x = self.norm1(x)  # 이미 view 및 permute 후 contiguous 처리됨
        x = x.contiguous()  # 추가로 한 번 호출
        x_input = x.contiguous().clone()
        attn_output, _ = self.attn(x_input, x_input, x_input)
        x = shortcut + attn_output
        x = x + self.mlp(self.norm2(x))

        x = x.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # (B, C, num_win_H, window_size, num_win_W, window_size)
        x = x.view(B, C, H, W)
        return x

# Swin Transformer Stack: 4번 반복
class SwinTransformerStack(nn.Module):
    def __init__(self, dim, num_blocks=4, **kwargs):
        super().__init__()
        blocks = []
        for _ in range(num_blocks):
            blocks.append(SwinTransformerBlock(dim, **kwargs))
        self.blocks = nn.Sequential(*blocks)
    def forward(self, x):
        return self.blocks(x)

##############################################
#  Improved Level1Waveblock: 단 한 번만 적용,
#  내부 detail branch에서 각 sub-band (LH, HL, HH)를 독립적으로 처리 + Fusion + Residual 연결
##############################################
class ImprovedLevel1Waveblock(nn.Module):
    def __init__(self, *, final_dim, window_size=7, num_heads=3, mlp_ratio=4.0, dropout=0.5):
        """
        final_dim: 입력 채널 수 (예: 128)
        - Reduction branch: 1x1 conv로 채널 수를 final_dim//4로 축소 (저주파)
        - Detail branch: 각 고주파 sub-band (LH, HL, HH)를 독립적으로 처리 후 fusion (고주파)
        """
        super().__init__()
        self.final_dim = final_dim
        self.reduction = nn.Conv2d(final_dim, final_dim // 4, kernel_size=1)
        self.dwt = DWTForward(J=1, mode='zero', wave='db1')
        # Projection for each high-frequency sub-band
        self.proj_LL = nn.Conv2d(final_dim // 4, final_dim // 4, kernel_size=1)
        self.proj_LH = nn.Conv2d(final_dim // 4, final_dim // 4, kernel_size=1)
        self.proj_HL = nn.Conv2d(final_dim // 4, final_dim // 4, kernel_size=1)
        self.proj_HH = nn.Conv2d(final_dim // 4, final_dim // 4, kernel_size=1)
        # 결정: 각 sub-band의 embed_dim = final_dim // 4가 num_heads와 나누어 떨어져야 함.
        # 만약 (final_dim//4) % num_heads != 0이면, 호환되는 값으로 대체(여기서는 2 사용)
        sub_band_dim = final_dim // 4
        sub_band_num_heads = num_heads if sub_band_dim % num_heads == 0 else 2

        # 독립적인 Swin Transformer Stack (각각 4블록)
        self.swin_stack_LL = SwinTransformerStack(dim=sub_band_dim, num_blocks=4,
                                                   window_size=window_size, num_heads=sub_band_num_heads,
                                                   mlp_ratio=mlp_ratio, dropout=dropout)
        self.swin_stack_LH = SwinTransformerStack(dim=sub_band_dim, num_blocks=4,
                                                   window_size=window_size, num_heads=sub_band_num_heads,
                                                   mlp_ratio=mlp_ratio, dropout=dropout)
        self.swin_stack_HL = SwinTransformerStack(dim=sub_band_dim, num_blocks=4,
                                                   window_size=window_size, num_heads=sub_band_num_heads,
                                                   mlp_ratio=mlp_ratio, dropout=dropout)
        self.swin_stack_HH = SwinTransformerStack(dim=sub_band_dim, num_blocks=4,
                                                   window_size=window_size, num_heads=sub_band_num_heads,
                                                   mlp_ratio=mlp_ratio, dropout=dropout)
        # Fusion: LL와 처리된 고주파 디테일을 결합
        self.conv_fusion = nn.Conv2d(final_dim, final_dim, kernel_size=1)
        self.upsample = nn.ConvTranspose2d(final_dim, final_dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # x: (B, final_dim, H, W)
        b, c, h, w = x.shape
        residual = x
        x_reduced = self.reduction(x)  # (B, final_dim//4, H, W)
        # 웨이브릿 분해: LL, LH, HL, HH 각각 (B, final_dim//4, H/2, W/2)
        LL, LH, HL, HH = self.dwt(x_reduced)
        # 각 고주파 sub-band에 대해 projection 및 독립적인 Transformer 처리
        LL_proj = self.proj_LL(LL)
        LH_proj = self.proj_LH(LH)
        HL_proj = self.proj_HL(HL)
        HH_proj = self.proj_HH(HH)
        LL_proc = self.swin_stack_LL(LL_proj) + LL_proj
        LH_proc = self.swin_stack_LH(LH_proj) + LH_proj  # branch 내 residual 연결
        HL_proc = self.swin_stack_HL(HL_proj) + HL_proj
        HH_proc = self.swin_stack_HH(HH_proj) + HH_proj
        # 처리된 고주파 정보들을 concat
        detail = torch.cat([LL_proc, LH_proc, HL_proc, HH_proc], dim=1)  # (B, 3*(final_dim//4), H/2, W/2)
        
        x_fused = self.conv_fusion(detail)  # (B, final_dim, H/2, W/2)
        out = self.upsample(x_fused)  # (B, final_dim, H, W)
        # 전체 residual 연결
        out = out + residual
        return out
##############################################
#  wavemix 관련 내부 함수 및 클래스 끝
##############################################

# 1. MaskPredictor: 손상 영역(마스크) 예측
class MaskPredictor(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32):
        super(MaskPredictor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(mid_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()  # 마스크 값은 0~1 사이
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        mask = self.sigmoid(self.conv3(x))
        return mask

# 2. WaveMixModule: Patch Embedding + ImprovedLevel1Waveblock (단 한 번 적용) + 후처리 및 디코더
class WaveMixModule(nn.Module):
    def __init__(self, *, depth, mult=2, ff_channel=16, final_dim=16, dropout=0.):
        super().__init__()
        # Patch Embedding: (B, 3, 224, 224) -> (B, final_dim, 112, 112)
        self.conv = nn.Sequential(
            nn.Conv2d(3, int(final_dim/2), 3, 1, 1),
            nn.Conv2d(int(final_dim/2), final_dim, 3, 2, 1),
        )
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # 이제 Level1Waveblock은 내부에 정의된 것으로 사용
            self.layers.append(ImprovedLevel1Waveblock(final_dim=final_dim,
                                                          window_size=7,
                                                          num_heads=3,
                                                          mlp_ratio=4.0,
                                                          dropout=dropout))
        # # ImprovedLevel1Waveblock 단 한 번 적용
        # self.improved_waveblock = ImprovedLevel1Waveblock(final_dim=final_dim,
        #                                                   window_size=7,
        #                                                   num_heads=3,
        #                                                   mlp_ratio=4.0,
        #                                                   dropout=dropout)
        self.depthconv = nn.Sequential(
            nn.Conv2d(final_dim, final_dim, 5, groups=final_dim, padding="same"),
            nn.GELU(),
            LayerNorm2d(final_dim)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(final_dim*2, int(final_dim/2), 4, stride=2, padding=1),
            LayerNorm2d(int(final_dim/2))
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(int(final_dim/2) + 3, 3, 1),
        )
    def forward(self, img, mask):
        # Patch Embedding
        x = self.conv(img)  # (B, final_dim, 112, 112)
        skip1 = x
        for attn in self.layers:
            x = attn(x) + x
        # # ImprovedLevel1Waveblock 단 한 번 적용 + residual 연결
        # x = self.improved_waveblock(x) + x
        x = self.depthconv(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.decoder1(x)
        x = torch.cat([x, img], dim=1)
        x = self.decoder2(x)
        return x

# 3. SelfInpaint: 마스크 예측 및 복원
class SelfInpaint(nn.Module):
    def __init__(self, *, num_modules=1, blocks_per_module=1, mult=4, ff_channel=16, final_dim=16, dropout=0.,
                 mask_predictor_mid_channels=32):
        super().__init__()
        self.mask_predictor = MaskPredictor(in_channels=3, mid_channels=mask_predictor_mid_channels)
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
        predicted_mask = self.mask_predictor(img)  # (B, 1, 224, 224)
        x = img
        for module in self.wavemodules:
            x = module(x, 1 - predicted_mask) + x
        inpainted = x * predicted_mask + img
        return inpainted
