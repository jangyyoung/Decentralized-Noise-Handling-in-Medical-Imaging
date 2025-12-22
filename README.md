# Decentralized-Noise-Handling-in-Medical-Imaging
Noise in medical imaging is an inevitable challenge, often
 stemming from acquisition artifacts, varying imaging protocols, and external interference. While some studies suggest that noise can enhance
 model robustness, excessive or unstructured noise degrades training qual
ity and classification performance. This issue is further exacerbated in
 federated learning settings, where individual clients have limited local
 data, making it difficult to train robust models independently. Federated
 imputation has been explored as a solution, yet existing methods do not
 fully leverage federated learning settings for optimal noise reconstruc
tion. In this work, we introduce a novel encoder-decoder based federated
 imputation method, designed to replace noisy images with more represen
tative reconstructions before training. Experimental results demonstrate
 that classification models trained with images imputed by the proposed
 method consistently outperform those trained with raw noisy images
 and without noisy images, highlighting the importance of effective noise
 handling in federated learning-based medical imaging.

This problem is further amplified in Federated Learning (FL) settings, where individual clients possess limited local data, making it difficult to independently train robust models. Although federated imputation has been explored as a potential solution, existing approaches do not fully exploit the federated learning paradigm for effective noise reconstruction.

In this work, we propose a novel encoder–decoder–based federated imputation method that replaces noisy images with more representative reconstructions prior to model training. Experimental results show that classification models trained using images imputed by the proposed method consistently outperform those trained on raw noisy images or datasets where noisy samples are removed, emphasizing the importance of effective noise handling in federated-learning-based medical imaging.

This repository presents a Federated Imputation framework designed to address noise in medical imaging within a decentralized learning environment.

This study proposes a novel encoder-decoder-based federated imputation method that replaces noisy medical images with high-quality reconstructions, significantly enhancing classification performance in decentralized learning environments.

<img src="fig 1.PNG" alt="architecture" width="800">
