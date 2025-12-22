# Decentralized Noise Handling in Medical Imaging

This repository introduces a novel **Federated Imputation** method designed to handle noise in medical imaging within a decentralized learning environment.

## 📌 Motivation
Noise in medical imaging is an inevitable challenge caused by acquisition artifacts and varying protocols. In **Federated Learning (FL)**, this issue is exacerbated because:
- Individual clients have limited local data.
- Standard models struggle to maintain robustness independently.
- Existing imputation methods don't fully leverage the federated setting.

## 🚀 Key Contributions
- **Novel Federated Imputation:** An encoder-decoder based method to reconstruct representative images from noisy data.
- **Preprocessing over Removal:** Instead of discarding noisy images, our method replaces them with high-quality reconstructions before training.
- **Enhanced Performance:** Classification models trained with our imputed images consistently outperform those trained with raw or filtered datasets.

## 🏗 System Architecture
The proposed method utilizes an encoder-decoder framework specifically optimized for federated reconstruction tasks.

<img src="fig 1.PNG" alt="architecture" width="800">

## 📊 Experimental Results
Our research demonstrates that effective noise handling in federated learning significantly improves classification accuracy compared to:
1. Training with raw noisy images.
2. Training without noisy images (data removal).

## 🛠 Tech Stack
- Federated Learning Framework
- Deep Learning (Encoder-Decoder Architecture)
- Medical Imaging Data Processing
