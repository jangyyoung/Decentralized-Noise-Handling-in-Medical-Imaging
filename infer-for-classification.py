import os
import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from one_model_LN_swin_v5 import SelfInpaint  # SelfInpaint 모델
import lpips
import torch.nn as nn
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

# 장치 설정 (CUDA 사용 가능하면 GPU 사용)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 모델 설정
NUM_MODULES = 8
NUM_BLOCKS = 4
MODEL_EMBEDDING = 128

model = SelfInpaint(
    num_modules=NUM_MODULES,
    blocks_per_module=NUM_BLOCKS,
    mult=4,
    ff_channel=MODEL_EMBEDDING,
    final_dim=MODEL_EMBEDDING,
    dropout=0.5,
    mask_predictor_mid_channels=32  # 마스크 예측기의 중간 채널 수
).to(device)

# 가중치 로드
model_path = "/home/prml/YY/2503_miccai/fed_inpaint/wavepaint_swin_v5.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Model weights loaded from {model_path}")
else:
    raise FileNotFoundError(f"❌ Model weights not found: {model_path}")

model.eval()

# 평가 지표 계산을 위한 함수 (옵션)
def eval_metrics(output, gt):
    metrics = {}
    metrics["L1"] = nn.L1Loss()(output, gt).mean().item()
    metrics["L2"] = nn.MSELoss()(output, gt).mean().item()
    metrics["PSNR"] = peak_signal_noise_ratio(output, gt).mean().item()
    metrics["SSIM"] = structural_similarity_index_measure(output, gt).mean().item()
    metrics["LPIPS"] = loss_fn(gt, output).mean().item()
    return metrics

# LPIPS 손실 함수 초기화 (평가지표 계산용)
loss_fn = lpips.LPIPS(net='alex').to(device)

# 클라이언트 데이터셋 경로 (각 클라이언트는 c0_dataset, c1_dataset, ... 폴더 내 존재)
data_dir = "/home/prml/YY/2503_miccai/datasets/train_datasets_non_iid"
num_clients = 10

# 각 클라이언트에 대해 infer 진행
for cid in range(num_clients):
    # train_wm 폴더 경로 (여기 metadata.csv가 존재)
    train_wm_dir = os.path.join(data_dir, f"c{cid}_dataset/train", "train_wm")
    metadata_csv_path = os.path.join(train_wm_dir, "metadata.csv")
    
    if not os.path.exists(metadata_csv_path):
        print(f"❌ metadata.csv not found in {train_wm_dir} for client c{cid}. Skipping client.")
        continue
    
    # metadata.csv 로드 (컬럼: image, mask, label)
    metadata_df = pd.read_csv(metadata_csv_path)
    print(f"✅ Loaded metadata.csv for client c{cid} from train_wm with {len(metadata_df)} entries.")
    
    # 출력 폴더: train_im (원본 이미지 이름으로 결과 이미지 저장)
    train_im_dir = os.path.join(data_dir, f"c{cid}_dataset", "train_im_swin_v5")
    os.makedirs(train_im_dir, exist_ok=True)
    
    # 새 metadata 기록을 위한 리스트
    new_metadata = []
    
    # 각 row마다 이미지, 마스크 로드 및 damaged 이미지 생성
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc=f"Client c{cid} Inference"):
        image_filename = row['image']
        mask_filename = row['mask']
        label = row['label']
        
        # 이미지와 마스크 파일 경로 (train_wm 폴더 내의 images와 masks 폴더)
        image_path = os.path.join(train_wm_dir, "images", image_filename)
        mask_path  = os.path.join(train_wm_dir, "masks", mask_filename)
        
        # 이미지 로드 (BGR -> RGB, 정규화, (C,H,W) 변환)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"❌ Image not found: {image_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # (3, H, W)
        
        # 마스크 로드 (grayscale, [0,1] 정규화, (1,H,W) 변환)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"❌ Mask not found: {mask_path}")
            continue
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)  # (1, H, W)
        
        # 손상된 이미지 생성: damaged = image * (1 - mask)
        damaged = image * (1 - mask)
        
        # torch tensor 변환 및 device 이동, 배치 차원 추가
        damaged_tensor = torch.tensor(damaged).unsqueeze(0).to(device).float()  # (1, 3, H, W)
        
        # 모델 추론
        with torch.no_grad():
            output_tensor = model(damaged_tensor)
        
        # 추론 결과 tensor를 numpy 이미지로 변환 (값 범위 [0,255])
        output_img = output_tensor.squeeze().cpu().permute(1, 2, 0).numpy() * 255
        output_img = np.clip(output_img, 0, 255).astype(np.uint8)
        
        # 결과 이미지 저장 (원본 이미지 파일명 사용)
        output_path = os.path.join(train_im_dir, image_filename)
        cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
        
        # 새 metadata에 기록 (파일명, label)
        new_metadata.append({"image": image_filename, "label": label})
    
    # 클라이언트별로 생성된 metadata.csv 저장 (train_im 폴더 내)
    new_metadata_df = pd.DataFrame(new_metadata)
    new_metadata_csv_path = os.path.join(train_im_dir, "metadata.csv")
    new_metadata_df.to_csv(new_metadata_csv_path, index=False)
    print(f"✅ Client c{cid} inference completed. Generated images and metadata saved in {train_im_dir}")

print("\nInference completed for all clients.")
