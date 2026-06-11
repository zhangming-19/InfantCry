# Enhancing infant cry recognition using lightweight CNN with hybrid feature augmentation

This project includes the source code for the paper [**Enhancing infant cry recognition using lightweight CNN with hybrid feature augmentation**](https://doi.org/10.1016/j.bspc.2026.110367), appearing at Biomedical Signal Processing and Control. Please cite this [article](https://doi.org/10.1016/j.bspc.2026.110367) as follows, if you use this code.

> Zhang M, Lu J, Cheng L, et al. Enhancing infant cry recognition using lightweight CNN with hybrid feature augmentation[J]. Biomedical Signal Processing and Control, 2026, 121: 110367.

## Requirements
We use Conda python 3.6 and strongly recommend that you create a new environment.
* Prerequisite: Python 3.6 or higher versions
```shell script
conda create -n MyEnv python=3.6
conda activate MyEnv
```

## Environment
This code is tested using Python 3.6, Pytorch 1.10, and CUDA 11.1
* Install all packages in the requirement.txt
```shell script
pip3 install -r requirements.txt
```

## Datasets
### Babycry
More details can be find in this [link](https://github.com/gveres/donateacry-corpus). please request and download the data from the original paper.


### Donateacry 
More details can be find in this [link](www.kaggle.com/datasets/chris0223/babycry). please request and download the data from the original paper.

## Codes
config.py

model.py

da.py

resnetAudio.py

## Citation
```
@article{ZHANG2026110367,
title = {Enhancing infant cry recognition using lightweight CNN with hybrid feature augmentation},
journal = {Biomedical Signal Processing and Control},
volume = {121},
pages = {110367},
year = {2026},
issn = {1746-8094},
doi = {https://doi.org/10.1016/j.bspc.2026.110367},
url = {https://www.sciencedirect.com/science/article/pii/S1746809426009213},
author = {Ming Zhang and Jiyu Lu and Lu Cheng and Xiancheng Yang and Jun Zhou and Meilin Wan},
keywords = {Infant cry, Lightweight network, Feature augmentation, Feature fusion, Log-Mel spectrogram},
abstract = {Automated infant cry classification remains challenging in real-world scenarios owing to the limited annotated data and the computational burden of deep learning models. To address these issues, this study proposes a lightweight convolutional neural network (CNNL) combined with a hybrid feature augmentation (HFA) strategy for robust infant cry recognition. The proposed CNNL contains only 6.3M parameters, substantially fewer than the 20.3M parameters of ResNet32, while achieving superior classification performance. Meanwhile, HFA enhances training diversity and model generalization through time shifting, speed variation, pitch transformation, and noise insertion. Experimental results on two benchmark datasets show that the proposed framework outperforms several representative baselines, including ResNet32, ResNet18, and MobileNetV2. With HFA, the proposed method achieves classification accuracies of 97.14% on the Babycry dataset and 95.00% on the Donateacry dataset. These results confirm the effectiveness of the proposed method as a compact and high-performing solution for infant cry classification.}
}
```

## Get Involved
Should you have any query please contact me at [zhangming@hccl.ioa.ac.cn](mailto:zhangming@hccl.ioa.ac.cn).
Please create a GitHub issue if you have any questions, suggestions, requests or bug-reports. 
Don't hesitate to send us an e-mail or report an issue, if something is broken or if you have further questions.
