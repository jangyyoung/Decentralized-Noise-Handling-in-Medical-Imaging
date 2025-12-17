import os
import sys
import time
import copy
import argparse
import numpy as np
from tqdm import tqdm
import gc  # 가비지 컬렉션

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import lpips

# Client 전용 데이터셋 로더 (분산된 train 데이터셋)
from datasets import make_client_train_dataloader
# 중앙집중식 데이터셋 로더 (통합된 validation, test 데이터셋)
from datasets import make_precomputed_val_dataloader, make_precomputed_test_dataloader

from one_model_LN_swin_v6 import SelfInpaint  # SelfInpaint 모델 사용

import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("-mask", "--mask", default="medium", help="Mask size: thick or medium")
parser.add_argument("-batch", "--Batch_size", default=8, help="Effective Batch Size")
# 분산된 train 데이터셋 폴더 (클라이언트 폴더가 c0_dataset, c1_dataset, … 형식)
parser.add_argument("-data", "--data_dir", default="/home/prml/YY/2503_miccai/datasets/train_datasets_non_iid", help="Path to distributed train dataset folder")
# 중앙집중식 검증 데이터셋 폴더 (구조: dataset/val/metadata.csv, images, masks)
parser.add_argument("-val", "--val_dir", default="/home/prml/YY/2503_miccai/datasets", help="Path to centralized validation dataset folder")
parser.add_argument("-g", "--global_rounds", default=10, help="Number of global rounds")
parser.add_argument("-l", "--local_epochs", default=1, help="Local training epochs per client")
args = parser.parse_args()

# 효과적으로 배치 8를 만들기 위해 micro_batch_size를 4로 설정하고
# 누적 스텝(accumulation_steps)은 8 / 4 = 2가 된다.
TrainBatchSize = int(args.Batch_size)      # 효과적 배치 크기 (예: 8)
micro_batch_size = 4                        # GPU에 올리는 실제 배치 크기 (메모리에 맞게 줄임)
accumulation_steps = TrainBatchSize // micro_batch_size

global_rounds = int(args.global_rounds)
local_epochs = int(args.local_epochs)
num_clients = 10  # 클라이언트 수 고정

# 첫번째 GPU를 기본 device로 지정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 평가에 사용할 LPIPS loss
loss_fn = lpips.LPIPS(net='alex').to(device)
out_size = 224  # MedMNIST 기본 이미지 사이즈

# HybridLoss 정의 (원본 코드와 동일)
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(HybridLoss, self).__init__()
        self.alpha = alpha

    def forward(self, x, y):
        l_lpips = loss_fn(x, y).mean()
        loss = l_lpips + (1 - self.alpha) * nn.L1Loss()(x, y) + (self.alpha * nn.MSELoss()(x, y))
        return loss * 10

criterion = HybridLoss().to(device)

# 평가 metric 함수 (LPIPS, L1, L2, PSNR, SSIM)
def EvalMetrics(out, gt):
    losses = {}
    losses["L1"] = nn.L1Loss()(out, gt).mean().item()
    losses["L2"] = nn.MSELoss()(out, gt).mean().item()
    losses["PSNR"] = peak_signal_noise_ratio(out, gt).mean().item()
    losses["SSIM"] = structural_similarity_index_measure(out, gt).mean().item()
    losses["LPIPS"] = loss_fn(gt, out).mean().item()
    return losses

def evaluate(model, dataloader):
    model.eval()
    Losses = {"L1": [], "L2": [], "PSNR": [], "SSIM": [], "LPIPS": []}
    with torch.no_grad():
        for data in dataloader:
            img = data["image"].to(device).float()      # 원본 이미지
            damaged = data["damaged"].to(device).float()  # 손상된 이미지
            output = model(damaged)
            metrics = EvalMetrics(output, img)
            for key in metrics:
                Losses[key].append(metrics[key])
    avg_losses = {k: np.mean(Losses[k]) for k in Losses}
    return avg_losses

# Federated Averaging: 각 클라이언트 모델의 파라미터를 데이터 수에 따라 가중평균
def federated_average(global_model, client_models, client_sample_counts):
    global_state = global_model.state_dict()
    total_samples = sum(client_sample_counts)
    for key in global_state.keys():
        global_state[key] = sum(
            client_models[i].state_dict()[key] * (client_sample_counts[i] / total_samples)
            for i in range(len(client_models))
        )
    global_model.load_state_dict(global_state)
    return global_model

# FedProx Aggregation (현재 FedProx의 글로벌 집계 단계는 FedAvg와 동일)
def fed_prox(global_model, client_models, client_sample_counts, mu=0):
    return federated_average(global_model, client_models, client_sample_counts)

# 글로벌 모델 초기화
NUM_MODULES = 8
NUM_BLOCKS = 2
MODEL_EMBEDDING = 128

global_model = SelfInpaint(
    num_modules=NUM_MODULES,
    blocks_per_module=NUM_BLOCKS,
    mult=4,
    ff_channel=MODEL_EMBEDDING,
    final_dim=MODEL_EMBEDDING,
    dropout=0.5,
    mask_predictor_mid_channels=32
).to(device)

summary(global_model, input_size=[(1, 3, out_size, out_size)], col_names=("input_size", "output_size", "num_params"), depth=6)

# 각 클라이언트별 분산 train 데이터 로더 구성
client_train_loaders = []
client_train_sizes = []
for cid in range(num_clients):
    # micro_batch_size로 데이터 로더 생성
    client_dir = os.path.join(args.data_dir, f"/home/prml/YY/2503_miccai/datasets/train_datasets_non_iid/c{cid}_dataset/train")
    train_loader = make_client_train_dataloader(client_dir, batch_size=micro_batch_size, num_workers=4)
    client_train_loaders.append(train_loader)
    client_train_sizes.append(len(train_loader.dataset))

# 중앙집중식 validation 데이터 로더 구성 (예: "dataset/val" 구조)
central_val_loader = make_precomputed_val_dataloader(base_dir=os.path.join(args.val_dir, "val"),
                                                     batch_size=TrainBatchSize,
                                                     num_workers=4)

print("Total training samples (across clients):", sum(client_train_sizes))
print("Total validation samples:", len(central_val_loader.dataset))

scaler = torch.amp.GradScaler('cuda:0')
optimizer = optim.AdamW(global_model.parameters(), lr=0.001, betas=(0.9, 0.999),
                          eps=1e-08, weight_decay=0.01, amsgrad=False)

global_round = 0
while global_round < global_rounds:
    print(f"\n=== Global Round {global_round+1}/{global_rounds} ===")
    client_models = []
    client_samples = []
    
    # 각 클라이언트 로컬 업데이트
    for cid in range(num_clients):
        print(f"\n--- Client {cid} training ---")
        local_model = copy.deepcopy(global_model)
        local_model.train()
        optimizer_local = optim.AdamW(local_model.parameters(), lr=0.001, betas=(0.9, 0.999),
                                        eps=1e-08, weight_decay=0.01)
        scaler_local = torch.amp.GradScaler('cuda:0')
        
        running_loss = 0.0
        total_batches = 0
        # local_epochs 단위로 누적: 각 에포크마다 그라디언트 누적을 위해 옵티마이저 zero_grad 호출
        for _ in range(local_epochs):
            optimizer_local.zero_grad()
            accumulation_counter = 0
            with tqdm(client_train_loaders[cid], unit="batch", desc=f"Client {cid} Epoch") as tepoch:
                for batch in tepoch:
                    image = batch["image"].to(device)
                    damaged = batch["damaged"].to(device)
                    target = image.clone()
                    
                    with torch.amp.autocast('cuda:0', enabled=False):
                        outputs = local_model(damaged)
                        loss = criterion(outputs, target)
                    # 누적 스텝에 맞게 loss를 나눠준다.
                    loss = loss / accumulation_steps
                    scaler_local.scale(loss).backward()
                    
                    accumulation_counter += 1
                    running_loss += loss.item() * accumulation_steps  # 원래 loss 값 복원
                    total_batches += 1
                    tepoch.set_postfix({"Loss": loss.item() * accumulation_steps})
                    
                    if accumulation_counter % accumulation_steps == 0:
                        scaler_local.step(optimizer_local)
                        scaler_local.update()
                        optimizer_local.zero_grad()
                        accumulation_counter = 0
                # 남은 gradient가 있다면 업데이트 수행
                if accumulation_counter != 0:
                    scaler_local.step(optimizer_local)
                    scaler_local.update()
                    optimizer_local.zero_grad()
        
        avg_local_loss = running_loss / total_batches if total_batches > 0 else 0
        print(f"Client {cid} average training loss: {avg_local_loss:.4f}")
        client_models.append(local_model)
        client_samples.append(len(client_train_loaders[cid].dataset))
        
        # 옵티마이저, 스케일러 등 사용 객체 명시적 삭제 및 메모리 해제
        del optimizer_local, scaler_local
        gc.collect()
        torch.cuda.empty_cache()
    
    # 모든 클라이언트의 로컬 모델을 집계하여 글로벌 모델 업데이트
    global_model = federated_average(global_model, client_models, client_samples)
    global_model = global_model.to(device)
    
    # 중앙집중식 validation 데이터셋으로 글로벌 모델 평가
    metrics = evaluate(global_model, central_val_loader)
    print(f"Global validation metrics: {metrics}")
    
    global_round += 1

# 모델 저장 (DataParallel 사용 시 module.state_dict() 사용)
state_dict = global_model.module.state_dict() if hasattr(global_model, "module") else global_model.state_dict()
save_path = f"go_blocks{NUM_BLOCKS}_dim{MODEL_EMBEDDING}_modules{NUM_MODULES}_federated_medmnist224.pth"
torch.save(state_dict, save_path)
print(f'Finished Federated Training, global model saved as {save_path}')
