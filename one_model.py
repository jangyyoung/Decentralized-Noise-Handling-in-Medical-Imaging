import torch
import torch.nn as nn
import wavemix
from wavemix import Level1Waveblock, Level2Waveblock, Level3Waveblock, DWTForward

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
    def __init__(
        self,
        *,
        depth,
        mult=2,
        ff_channel=16,
        final_dim=16,
        dropout=0.,
    ):
        super().__init__()
        
        # 원래 이미지 채널(3)을 입력으로 받도록 구성
        self.conv = nn.Sequential(
            nn.Conv2d(3, int(final_dim/2), 3, 1, 1),
            nn.Conv2d(int(final_dim/2), final_dim, 3, 2, 1),
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Level1Waveblock(mult=mult, ff_channel=ff_channel, final_dim=final_dim, dropout=dropout))

        self.depthconv = nn.Sequential(
            nn.Conv2d(final_dim, final_dim, 5, groups=final_dim, padding="same"),
            nn.GELU(),
            nn.BatchNorm2d(final_dim)
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(final_dim*2, int(final_dim/2), 4, stride=2, padding=1),
            nn.BatchNorm2d(int(final_dim/2))
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(int(final_dim/2) + 3, 3, 1),
        )
        
    def forward(self, img, mask):
        # (원래 코드는 torch.cat([img, mask], dim=1)로 시작하지만 실제로 사용되지는 않음)
        # x = torch.cat([img, mask], dim=1)
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
    def __init__(
        self,
        *,
        num_modules=1,
        blocks_per_module=7,
        mult=4,
        ff_channel=16,
        final_dim=16,
        dropout=0.,
        mask_predictor_mid_channels=32
    ):
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
        # 여기서 predicted_mask는 0~1 사이 값이며, 1에 가까울수록 손상 영역으로 간주

        # Step 2: 기존 WaveMixModule을 이용한 복원 과정
        x = img
        # WaveMixModule에는 마스크의 보완(1 - predicted_mask)을 전달
        for module in self.wavemodules:
            x = module(x, 1 - predicted_mask) + x
        # 최종 출력은 예측한 마스크 영역은 모델 복원 결과, 나머지 영역은 원본 이미지를 유지
        inpainted = x * predicted_mask + img
        return inpainted
