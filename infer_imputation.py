import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

# 모델 관련 모듈 임포트 (SelfInpaint 정의가 들어있는 파일)
from one_model_LN_swin_v4 import SelfInpaint

# (필요에 따라) 학습 시 사용한 하이퍼파라미터와 동일하게 설정
NUM_MODULES = 8
NUM_BLOCKS = 4
MODEL_EMBEDDING = 128

# 장치 설정 (GPU 우선)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 글로벌 모델 생성 및 가중치 로드
global_model = SelfInpaint(
    num_modules=NUM_MODULES,
    blocks_per_module=NUM_BLOCKS,
    mult=4,
    ff_channel=MODEL_EMBEDDING,
    final_dim=MODEL_EMBEDDING,
    dropout=0.5,
    mask_predictor_mid_channels=32
).to(device)

# DataParallel로 학습한 경우
if hasattr(global_model, "module"):
    state_dict = torch.load("/home/prml/YY/2503_miccai/fed_inpaint/wavepaint_swin.pth".format(
        NUM_BLOCKS, MODEL_EMBEDDING, NUM_MODULES), map_location=device)
    global_model.module.load_state_dict(state_dict)
else:
    state_dict = torch.load("/home/prml/YY/2503_miccai/fed_inpaint/wavepaint_swin.pth".format(
        NUM_BLOCKS, MODEL_EMBEDDING, NUM_MODULES), map_location=device)
    global_model.load_state_dict(state_dict)

global_model.eval()  # 추론 모드로 전환

# 클라이언트 데이터 및 결과 저장 관련 경로 설정
# 분산된 클라이언트 데이터셋이 있는 기본 폴더 (예: "distribute_datasets_non_iid")
client_base_dir = "/home/prml/YY/2503_miccai/datasets/train_datasets_non_iid"
# 최종 결과를 저장할 output 기본 폴더
output_base_dir = "/home/prml/YY/2503_miccai/datasets/train_datasets_non_iid/output_swin"

# 클라이언트 수 (예: 10)
num_clients = 10

# 각 클라이언트별로 infer 수행
for cid in range(num_clients):
    print(f"\n=== Client {cid} infer ===")
    
    # 클라이언트 데이터 경로 (예: distribute_datasets_non_iid/c0_dataset)
    client_dir = os.path.join(client_base_dir, f"c{cid}_dataset")
    # 입력 폴더: train/train_wm 안에 images와 masks 폴더가 있다고 가정
    train_wm_dir = os.path.join(client_dir, "train", "train_wm")
    images_dir = os.path.join(train_wm_dir, "images")
    masks_dir  = os.path.join(train_wm_dir, "masks")
    
    # 출력 폴더: output/<client>/train/train_im (없으면 생성)
    output_dir = os.path.join(output_base_dir, f"c{cid}_dataset", "train", "train_im")
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 파일 목록 (예: png, jpg 등)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_name in tqdm(image_files, desc=f"Client {cid} Inference"):
        img_path = os.path.join(images_dir, img_name)
        mask_path = os.path.join(masks_dir, img_name)  # 파일명이 동일하다고 가정
        
        # 이미지 로드 (cv2는 기본 BGR로 읽으므로, RGB 변환)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"❌ 이미지 로드 실패: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype(np.float32) / 255.0  # [0,1] 스케일
        
        # 마스크 로드 (마스크는 단일 채널로 로드)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"❌ 마스크 로드 실패: {mask_path}")
            continue
        mask = mask.astype(np.float32) / 255.0  # [0,1] 스케일
        
        # 채널 맞추기: 이미지는 (H, W, 3), 마스크는 (H, W) -> (H, W, 1) 후 반복
        mask = np.expand_dims(mask, axis=-1)
        mask = np.repeat(mask, 3, axis=-1)  # 3채널로 확장
        
        # 손상된 이미지 생성: 여기서는 원본 이미지에서 마스크 영역을 제거 (1 - mask)
        damaged_rgb = img_rgb * (1 - mask)
        
        # 텐서 변환 및 차원 변경 (H,W,C) -> (1,C,H,W)
        damaged_tensor = torch.from_numpy(damaged_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # 글로벌 모델 추론
        with torch.no_grad():
            output_tensor = global_model(damaged_tensor)
        
        # 결과 텐서를 이미지 형식으로 변환 (값을 0~255 범위로 복원)
        output_img = output_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        output_img = np.clip(output_img, 0, 1) * 255.0
        output_img = output_img.astype(np.uint8)
        # 다시 BGR로 변환 (cv2.imwrite 사용)
        output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        
        # 결과 저장 경로
        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, output_bgr)
        
    print(f"✅ Client {cid} infer 완료. 결과는 {output_dir} 에 저장됨.")
