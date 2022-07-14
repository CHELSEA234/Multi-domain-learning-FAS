# Multi-domain Learning for Updating Face Anti-spoofing Models
This is the official code for our ECCV2022 oral paper "Multi-domain Learning for Updating Face Anti-spoofing Models".

[Xiao Guo](https://scholar.google.com/citations?user=Gkc-lAEAAAAJ&hl=en), [Yaojie Liu](https://yaojieliu.github.io/), [Anil Jain](https://www.cse.msu.edu/~jain/), [Xiaoming Liu](https://www.cse.msu.edu/~liuxm/index2.html)

<p align="center">
<img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/figures/overall_architecture.jpg" alt="drawing" width="800"/>
</p>

## Prerequisites
- Python 3.6 - 3.8
- Tensorflow 2.3.0 - 2.7.0
- Numpy 1.18.5
- opencv-python 4.5.2.54

## Dataset
The FASMD Dataset is constructed on  SiW-Mv2, SiW, and Oulu-NPU. It consists of five sub-datasets: dataset A is the
source domain dataset, and B, C, D and E are four target domain datasets. The details can be found in [dataset](https://github.com/CHELSEA234/Multi-domain-learning-FAS/tree/main/dataset)

<p align="center">
<img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/figures/Dataset_demo.png" alt="drawing" width="800"/>
<img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/figures/distribution.png" alt="drawing" width="800"/>
</p>

## Train and Inference
- After setting up the dataset path, you can run the training code as shown below:

```
    python train_architecture.py
```
- To run the testing code, which will save scores in csv file.
```
    python test_architecture.py
    python csv_parser.py
```

## Pre-trained model
The pre-trained model can be found in [link](https://drive.google.com/drive/folders/1CHIzOUyy3YvpDi-gP6nCIdOPHJWWxQQo?usp=sharing)

## Reference
If you would like to use our work, please cite:
```
@inproceedings{xiaoguo2023MDFAS
      title={Multi-domain Learning for Updating Face Anti-spoofing Models}, 
      author={Xiao, Guo and Yaojie, Liu, Anil, Jain and Liu, Xiaoming},
      booktitle={In Proceeding of European Conference on Computer Vision (ECCV 2022)},
      year={2022}
      
}
```
If you have any question, please contact: [Xiao Guo](guoxia11@msu.edu) 
