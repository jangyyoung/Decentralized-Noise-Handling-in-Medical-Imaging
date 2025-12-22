Noise in medical imaging is an unavoidable challenge, commonly arising from acquisition artifacts, heterogeneous imaging protocols, and external interference. While some studies suggest that noise may improve model robustness, excessive or unstructured noise can significantly degrade training quality and classification performance.

This problem is further amplified in Federated Learning (FL) settings, where individual clients possess limited local data, making it difficult to independently train robust models. Although federated imputation has been explored as a potential solution, existing approaches do not fully exploit the federated learning paradigm for effective noise reconstruction.

In this work, we propose a novel encoder–decoder–based federated imputation method that replaces noisy images with more representative reconstructions prior to model training. Experimental results show that classification models trained using images imputed by the proposed method consistently outperform those trained on raw noisy images or datasets where noisy samples are removed, emphasizing the importance of effective noise handling in federated-learning-based medical imaging.

This repository presents a Federated Imputation framework designed to address noise in medical imaging within a decentralized learning environment.
