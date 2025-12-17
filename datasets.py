import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import albumentations as A

##############################################
# 클라이언트용 데이터셋 (분산 데이터셋)
##############################################
class ClientMedMNISTDataset(Dataset):
    def __init__(self, client_split_dir, transform=None):
        """
        client_split_dir: 클라이언트 데이터셋 폴더 경로 (예: "distribute_datasets/c0_dataset/train")
        transform: 이미지 및 마스크에 적용할 Albumentations transform (선택 사항)
        """
        self.client_split_dir = client_split_dir
        self.transform = transform
        
        # images, masks 폴더 및 metadata.csv 경로 설정
        self.images_dir = os.path.join(client_split_dir, "images")
        self.masks_dir = os.path.join(client_split_dir, "masks")
        self.metadata_path = os.path.join(client_split_dir, "metadata.csv")
        
        # metadata CSV 로드 (예: 컬럼: image, mask, label)
        self.metadata = pd.read_csv(self.metadata_path)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_filename = row['image']
        mask_filename = row['mask']
        label = row['label']  # 필요 시 사용
        
        # 이미지 로드 및 전처리 (BGR -> RGB, 정규화, (C,H,W) 변환)
        image_path = os.path.join(self.images_dir, image_filename)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        # 마스크 로드 (grayscale, [0,1] 정규화, (1,H,W) 변환)
        mask_path = os.path.join(self.masks_dir, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)
        
        # 손상된 이미지: 원본 이미지에서 마스크 영역을 제거한 결과
        damaged = image * (1 - mask)
        
        sample = {
            "image": image,
            "mask": mask,
            "damaged": damaged,
            "label": label
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

def make_client_train_dataloader(client_dir, batch_size=32, num_workers=4, transform=None):
    """
    client_dir: 클라이언트 데이터셋 폴더 경로 (예: "distribute_datasets/c0_dataset")
    내부의 "train" 폴더에서 데이터를 로드하여 DataLoader 반환
    """
    train_dataset = ClientMedMNISTDataset(os.path.join(client_dir, "train_nm"), transform=transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


def make_client_train_wm_dataloader(client_dir, batch_size=32, num_workers=4, transform=None):
    """
    client_dir: 클라이언트 데이터셋 폴더 경로 (예: "distribute_datasets/c0_dataset")
    내부의 "train" 폴더에서 데이터를 로드하여 DataLoader 반환
    """
    train_dataset = ClientMedMNISTDataset(os.path.join(client_dir, "train_wm"), transform=transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)




##############################################
# 중앙집중식 데이터셋 (Precomputed 데이터셋)
##############################################
class PrecomputedMedMNISTDataset(Dataset):
    def __init__(self, base_dir, split='train', transform=None):
        """
        base_dir: 최상위 데이터셋 폴더 (예: "dataset" 또는 "dataset/val", "dataset/test")
        split: "train", "val", 또는 "test". 만약 base_dir에 이미 metadata.csv가 존재하면 split을 무시합니다.
        transform: 이미지 및 마스크에 적용할 Albumentations transform (선택 사항)
        """
        self.transform = transform
        
        # 만약 base_dir에 이미 metadata.csv가 있다면, base_dir은 최종 폴더입니다.
        if os.path.exists(os.path.join(base_dir, "metadata.csv")):
            self.base_dir = base_dir
            self.split = None
            self.images_dir = os.path.join(base_dir, "images")
            self.masks_dir = os.path.join(base_dir, "masks")
            self.metadata_path = os.path.join(base_dir, "metadata.csv")
        else:
            self.base_dir = base_dir
            self.split = split
            self.images_dir = os.path.join(base_dir, split, "images")
            self.masks_dir = os.path.join(base_dir, split, "masks")
            self.metadata_path = os.path.join(base_dir, split, "metadata.csv")
            
        # 메타데이터 CSV 로드 (예: 컬럼: image, mask, label)
        self.metadata = pd.read_csv(self.metadata_path)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_filename = row['image']
        mask_filename = row['mask']
        label = row['label']
        
        # 이미지 경로 구성 및 로드 (BGR -> RGB, 정규화, (C,H,W) 변환)
        image_path = os.path.join(self.images_dir, image_filename)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        # 마스크 경로 구성 및 로드 (grayscale, [0,1] 정규화, (1,H,W) 변환)
        mask_path = os.path.join(self.masks_dir, mask_filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)
        
        # 손상된 이미지: 원본 이미지에서 마스크 영역을 제거한 결과
        damaged = image * (1 - mask)
        
        sample = {
            "image": image,
            "mask": mask,
            "damaged": damaged,
            "label": label
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

def make_precomputed_train_dataloader(base_dir, batch_size=32, num_workers=4, transform=None):
    dataset = PrecomputedMedMNISTDataset(base_dir, split='train', transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

def make_precomputed_val_dataloader(base_dir, batch_size=32, num_workers=4, transform=None):
    dataset = PrecomputedMedMNISTDataset(base_dir, split='val', transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

def make_precomputed_test_dataloader(base_dir, batch_size=32, num_workers=4, transform=None):
    dataset = PrecomputedMedMNISTDataset(base_dir, split='test', transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
