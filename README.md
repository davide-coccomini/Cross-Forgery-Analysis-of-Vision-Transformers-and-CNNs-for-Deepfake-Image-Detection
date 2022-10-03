# Cross-Forgery-Analysis-of-Vision-Transformers-and-CNNs-for-Deepfake-Image-Detection

## Code from the paper: https://dl.acm.org/doi/abs/10.1145/3512732.3533582

Deepfake Generation Techniques are evolving at a rapid pace, making it possible to create realistic manipulated images and videos and endangering the serenity of modern society. The continual emergence of new and varied techniques brings with it a further problem to be faced, namely the ability of deepfake detection models to update themselves promptly in order to be able to identify manipulations carried out using even the most recent methods. This is an extremely complex problem to solve, as training a model requires large amounts of data, which are difficult to obtain if the deepfake generation method is too recent. Moreover, continuously retraining a network would be unfeasible. In this paper, we ask ourselves if, among the various deep learning techniques, there is one that is able to generalise the concept of deepfake to such an extent that it does not remain tied to one or more specific deepfake generation methods used in the training set. We compared a Vision Transformer with an EfficientNetV2 on a cross-forgery context based on the ForgeryNet dataset. From our experiments, It emerges that EfficientNetV2 has a greater tendency to specialize often obtaining better results on training methods while Vision Transformers exhibit a superior generalization ability that makes them more competent even on images generated with new methodologies.

# Reference
```
@inproceedings{10.1145/3512732.3533582,
author = {Coccomini, Davide Alessandro and Caldelli, Roberto and Falchi, Fabrizio and Gennaro, Claudio and Amato, Giuseppe},
title = {Cross-Forgery Analysis of Vision Transformers and CNNs for Deepfake Image Detection},
year = {2022},
isbn = {9781450392426},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3512732.3533582},
doi = {10.1145/3512732.3533582},
booktitle = {Proceedings of the 1st International Workshop on Multimedia AI against Disinformation},
pages = {52–58},
numpages = {7},
keywords = {deep fake detection, transformer networks, deep learning},
location = {Newark, NJ, USA},
series = {MAD '22}
}


```