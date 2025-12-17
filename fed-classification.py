import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
from tqdm import tqdm
import numpy as np
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# ---------------------------
# Custom Dataset for Client Training
# ---------------------------
class CustomClientDataset(Dataset):
    """
    Loads a client's training data from the 'train_nm' folder.
    Expects a metadata.csv with at least two columns: 'image' and 'label'.
    """
    def __init__(self, folder, transform=None):
        self.folder = folder
        metadata_path = os.path.join(folder, "metadata.csv")
        self.metadata = __import__('pandas').read_csv(metadata_path)
        self.images_dir = os.path.join(folder, "images")
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.images_dir, row['image'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        # Assume label is an integer; adjust as necessary
        label = int(row['label'])
        return image, label
    
class CustomClientMaskedDataset(Dataset):
    """
    Loads a client's training data from the 'train_nm' folder.
    Expects a metadata.csv with at least two columns: 'image' and 'label'.
    """
    def __init__(self, folder, transform=None):
        self.folder = folder
        metadata_path = os.path.join(folder, "metadata.csv")
        self.metadata = __import__('pandas').read_csv(metadata_path)
        self.images_dir = os.path.join(folder, "images")
        self.masks_dir = os.path.join(folder, "masks")
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.images_dir, row['image'])
        mask_path = os.path.join(self.masks_dir, row['mask'])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = ImageOps.invert(mask)
        
        image = self.transform(image)
        mask = self.transform(mask)
        combined = image * mask
        label = int(row['label'])
        return combined, label
    
class CustomClientImputedDataset(Dataset):
    """
    Loads a client's training data from the 'train_nm' folder.
    Expects a metadata.csv with at least two columns: 'image' and 'label'.
    """
    def __init__(self, folder, transform=None):
        self.folder = folder
        metadata_path = os.path.join(folder, "metadata.csv")
        self.metadata = __import__('pandas').read_csv(metadata_path)
        self.images_dir = os.path.join(folder)
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.images_dir, row['image'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        # Assume label is an integer; adjust as necessary
        label = int(row['label'])
        return image, label

# ---------------------------
# Global Dataset for Validation/Test
# ---------------------------
class GlobalDataset(Dataset):
    """
    Loads images and labels from a global folder (val or test).
    Assumes the folder structure contains:
       - images/ folder
       - metadata.csv (with 'image' and 'label' columns)
    """
    def __init__(self, folder, transform=None):
        self.folder = folder
        metadata_path = os.path.join(folder, "metadata.csv")
        self.metadata = __import__('pandas').read_csv(metadata_path)
        self.images_dir = os.path.join(folder, "images")
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.images_dir, row['image'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = int(row['label'])
        return image, label

# ---------------------------
# Federated Learning Helper: FedAvg Aggregation
# ---------------------------
def federated_average(state_dicts):
    """Averages a list of model state_dicts (FedAvg)."""
    global_state = copy.deepcopy(state_dicts[0])
    for key in global_state.keys():
        for i in range(1, len(state_dicts)):
            global_state[key] += state_dicts[i][key]
        global_state[key] = global_state[key] / len(state_dicts)
    return global_state

# ---------------------------
# Federated Learning Pipeline
# ---------------------------
from classification import classification
from torch.utils.data import ConcatDataset
def federated_train(num_clients=10, num_rounds=10, local_epochs=3, batch_size=32, num_classes=10):
    """
    Federated training using the provided classification model.
      - Each client uses training data from its train_nm folder.
      - Global validation and test sets are loaded from dataset/val and dataset/test.
      - Uses FedAvg to aggregate model updates.
    """
    # Import your classification pipeline and CNN model


    # Initialize the global model
    global_model = classification(n_channels=3, n_classes=num_classes,
                                  task="classification", learning_rate=0.001, name="global")
    global_state = global_model.model.state_dict()

    # Set up global validation and test loaders
    val_dataset = GlobalDataset(os.path.join("/home/prml/YY/dataset", "val"), transform=transforms.ToTensor())
    test_dataset = GlobalDataset(os.path.join("/home/prml/YY/dataset", "test"), transform=transforms.ToTensor())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for round_idx in range(num_rounds):
        print(f"\n=== Global Round {round_idx+1}/{num_rounds} ===")
        local_states = []
 
        for client_idx in range(num_clients):
            print(f"\n--- Client {client_idx} training ---")
            
            client_train_nm = os.path.join("/home/prml/YY/2503_miccai/datasets/train_datasets_non_iid",f"c{client_idx}_dataset", "train", "train_nm")
            client_train_im = os.path.join("/home/prml/YY/2503_miccai/datasets/train_datasets_non_iid",f"c{client_idx}_dataset","train", "train_wm")
            #client_train_im = os.path.join("/home/prml/YY/2503_miccai/datasets/train_datasets_non_iid",f"c{client_idx}_dataset","train_im_swin_v5")
            client_nm_dataset = CustomClientDataset(client_train_nm, transform=transforms.ToTensor())
            client_im_dataset = CustomClientDataset(client_train_im, transform=transforms.ToTensor())
            #client_im_dataset = CustomClientImputedDataset(client_train_im, transform=transforms.ToTensor())
            client_dataset =ConcatDataset([client_nm_dataset,client_im_dataset])
            
            # client_train_nm = os.path.join("distribute_datasets_non_iid",f"c{client_idx}_dataset", "train", "train_nm")
            # client_train_wm = os.path.join("distribute_datasets_non_iid",f"c{client_idx}_dataset", "train", "train_wm")
            # client_nm_dataset = CustomClientDataset(client_train_nm, transform=transforms.ToTensor())
            # client_wm_dataset = CustomClientMaskedDataset(client_train_wm, transform=transforms.ToTensor())
            # client_dataset =ConcatDataset([client_nm_dataset,client_wm_dataset])
            
            # client_train_folder = os.path.join("distribute_datasets_non_iid",f"c{client_idx}_dataset", "train", "train_nm")
            # client_dataset = CustomClientDataset(client_train_folder, transform=transforms.ToTensor())
            client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)

            # Initialize local model with the global state
            local_model = classification(n_channels=3, n_classes=num_classes,
                                         task="classification", learning_rate=0.001,
                                         name=f"client_{client_idx}")
            local_model.model.load_state_dict(global_state)

            # Train locally for the specified number of epochs
            local_model.train(train_loader=client_loader, val_loader=val_loader, epochs=local_epochs)
            local_states.append(copy.deepcopy(local_model.model.state_dict()))

        # Federated averaging: update the global model
        global_state = federated_average(local_states)
        global_model.model.load_state_dict(global_state)

        # Evaluate the updated global model on the global validation set
        val_acc, val_auc = global_model.test(val_loader, display_confusion_matrix=False)
        print(f"Round {round_idx+1}: Global validation accuracy: {val_acc}, AUC: {val_auc}")

    # Final evaluation on the global test set
    print("\nFinal Evaluation on Global Test Set:")
    global_model.test(test_loader, display_confusion_matrix=True)

# ---------------------------
# Example: Run Federated Learning Pipeline
# ---------------------------
if __name__ == "__main__":
    # You may need to adjust num_classes and batch_size as appropriate
    federated_train(num_clients=10, num_rounds=10, local_epochs=1, batch_size=32, num_classes=9)
