import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from datasets import make_precomputed_test_dataloader  # test split 불러오기
from model import WavePaint  # SelfInpaint 모델 사용
import lpips
import torch.nn as nn
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

# FID 계산에 필요한 InceptionV3와 sqrtm 임포트
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import torch.nn.functional as F

# 장치 설정 (CUDA 사용 가능하면 GPU 사용)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 모델 설정
NUM_MODULES = 8
NUM_BLOCKS = 4
MODEL_EMBEDDING = 128

model = WavePaint(
    num_modules=NUM_MODULES,
    blocks_per_module=NUM_BLOCKS,
    mult=4,
    ff_channel=MODEL_EMBEDDING,
    final_dim=MODEL_EMBEDDING,
    dropout=0.5,
).to(device)

# 가중치 로드
model_path = "/home/prml/YY/2503_miccai/fed_inpaint/Cen_WavePaint_Base_B4.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Model weights loaded from {model_path}")
else:
    raise FileNotFoundError(f"❌ Model weights not found: {model_path}")

model.eval()

# LPIPS 손실 함수 초기화 (평가지표 계산용)
loss_fn = lpips.LPIPS(net='alex').to(device)

# 미리 저장된 test 데이터셋 로드 (base_dir: "dataset" 폴더 내의 test split)
dataloader = make_precomputed_test_dataloader(base_dir="/home/prml/YY/2503_miccai/datasets", batch_size=1, num_workers=4)
print(f"✅ Loaded {len(dataloader.dataset)} test samples.")

# 출력 폴더 설정
output_dir = "Cen_WavePaint_Base_B4/test_generated"
damaged_dir = "Cen_WavePaint_Base_B4/test_damaged"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(damaged_dir, exist_ok=True)

# 평가 지표 계산 함수 (LPIPS, L1, L2, PSNR, SSIM)
def eval_metrics(output, gt):
    metrics = {}
    metrics["L1"] = nn.L1Loss()(output, gt).mean().item()
    metrics["L2"] = nn.MSELoss()(output, gt).mean().item()
    metrics["PSNR"] = peak_signal_noise_ratio(output, gt).mean().item()
    metrics["SSIM"] = structural_similarity_index_measure(output, gt).mean().item()
    metrics["LPIPS"] = loss_fn(gt, output).mean().item()
    return metrics

# --- FID 계산 함수 ---
def compute_fid(features1, features2):
    # features1, features2: list of (N, dims) numpy arrays
    features1 = np.concatenate(features1, axis=0)
    features2 = np.concatenate(features2, axis=0)
    mu1 = np.mean(features1, axis=0)
    sigma1 = np.cov(features1, rowvar=False)
    mu2 = np.mean(features2, axis=0)
    sigma2 = np.cov(features2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def get_inception_features(x, model, device):
    # x: (B, C, H, W) 값은 [0,1]
    # InceptionV3는 299x299 입력을 요구하므로 리사이즈합니다.
    x = F.interpolate(x, size=(299,299), mode='bilinear', align_corners=False)
    with torch.no_grad():
        pred = model(x)
    # InceptionV3의 마지막 pooling layer의 출력을 사용합니다.
    if isinstance(pred, tuple):
        pred = pred[0]
    return pred.detach().cpu().numpy()

# Inception 모델 로드 (FID 계산용)
inception_model = inception_v3(pretrained=True, transform_input=False)
inception_model.eval()
inception_model.to(device)

# 전체 테스트 데이터셋에 대한 누적 평가 지표
total_metrics = {"L1": 0, "L2": 0, "PSNR": 0, "SSIM": 0, "LPIPS": 0}
sample_count = 0

# FID 계산을 위한 특징 저장 리스트
all_features_real = []
all_features_fake = []

# 테스트 데이터셋에 대한 Inpainting 수행, 결과 저장 및 평가 지표 누적 계산
for idx, data in enumerate(tqdm(dataloader, total=len(dataloader), desc="Running Test Inference")):
    # precomputed 데이터셋은 "image", "mask", "damaged", "label" 키를 포함합니다.
    damaged_img = torch.tensor(data["damaged"]).to(device).float()  # (B, C, H, W)
    mask= torch.tensor(data["mask"]).to(device).float()  # (B, 1, H, W)
    gt_img = torch.tensor(data["image"]).to(device).float()           # 원본 이미지

    with torch.no_grad():
        output_img = model(damaged_img,mask)

    # 평가 지표 계산 (각 샘플별)
    metrics = eval_metrics(output_img, gt_img)
    for key in total_metrics.keys():
        total_metrics[key] += metrics[key]
    sample_count += 1

    # Inception 특징 추출 (배치 크기가 1이므로 각 이미지별 특징)
    feat_real = get_inception_features(gt_img, inception_model, device)
    feat_fake = get_inception_features(output_img, inception_model, device)
    all_features_real.append(feat_real)
    all_features_fake.append(feat_fake)

    # 결과 이미지 저장: 파일명은 "image_{idx}.png"
    img_name = f"image_{idx}.png"
    output_path = os.path.join(output_dir, img_name)
    damaged_path = os.path.join(damaged_dir, img_name)

    # 텐서를 numpy 이미지로 변환 (값 범위 [0, 255])
    output_np = output_img.squeeze().cpu().permute(1, 2, 0).numpy() * 255
    damaged_np = damaged_img.squeeze().cpu().permute(1, 2, 0).numpy() * 255

    cv2.imwrite(output_path, cv2.cvtColor(output_np.astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(damaged_path, cv2.cvtColor(damaged_np.astype(np.uint8), cv2.COLOR_RGB2BGR))

# 전체 테스트 데이터셋에 대한 평균 평가 지표 계산
avg_metrics = {k: total_metrics[k] / sample_count for k in total_metrics.keys()}

# FID 계산 (모든 특징 모아서 계산)
fid_score = compute_fid(all_features_real, all_features_fake)
avg_metrics["FID"] = fid_score
avg_metrics['Hybrid Loss'] = (avg_metrics['LPIPS'] + (0.5*avg_metrics["L1"]) + (0.5*avg_metrics["L2"]))*10
print("✅ Test inference completed.")
print("Final Average Metrics on Test Set:")
for metric, value in avg_metrics.items():
    print(f"{metric}: {value:.4f}")

# 모델 파라미터 개수 출력
total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params}")

print(f"Results saved in {output_dir} and {damaged_dir}")
