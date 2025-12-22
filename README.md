# Decentralized-Noise-Handling-in-Medical-Imaging

## Abstract
Noise in medical imaging is an inevitable challenge, often stemming from acquisition artifacts, varying imaging protocols, and external interference. While some studies suggest that noise can enhance model robustness, excessive or unstructured noise degrades training quality and classification performance. 

This issue is further exacerbated in **federated learning (FL)** settings, where individual clients have limited local data, making it difficult to train robust models independently. Federated imputation has been explored as a solution, yet existing methods do not fully leverage federated learning settings for optimal noise reconstruction. 

In this work, we introduce a novel **encoder-decoder-based federated imputation method**, designed to replace noisy images with more representative reconstructions before training. Experimental results demonstrate that classification models trained with images imputed by the proposed method consistently outperform those trained with raw noisy images and without noisy images, highlighting the importance of effective noise handling in federated-learning-based medical imaging.

 <img src="fig 1.PNG" alt="architecture">
