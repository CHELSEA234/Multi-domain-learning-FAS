# Multi-domain Learning for Updating Face Anti-spoofing Models

<p align="center">
<img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/source_multi_domain/figures/overall_architecture.jpg" alt="drawing" width="1000"/>
</p>

This page contains the official implementation of our ECCV2022 oral paper "Multi-domain Learning for Updating Face Anti-spoofing Models". [[Arxiv]](https://arxiv.org/pdf/2208.11148.pdf) [[SiW-Mv2 Dataset]](http://cvlab.cse.msu.edu/pdfs/guo_liu_jain_liu_eccv2022_supp.pdf) 

**Our algorithm has been officially accepted and delivered to the [IAPRA ODIN](https://www.iarpa.gov/research-programs/odin) program**!

Authors: [Xiao Guo](https://scholar.google.com/citations?user=Gkc-lAEAAAAJ&hl=en), [Yaojie Liu](https://yaojieliu.github.io/), [Anil Jain](https://www.cse.msu.edu/~jain/), [Xiaoming Liu](http://cvlab.cse.msu.edu/)

## Dataset
The FASMD Dataset is constructed on  SiW-Mv2, SiW, and Oulu-NPU. It consists of five sub-datasets: dataset A is the
source domain dataset, and B, C, D and E are four target domain datasets. 
<p align="center">
    <img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/source_multi_domain/figures/Dataset_demo.png" alt="drawing" width="800"/>
    <img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/source_multi_domain/figures/distribution.png" alt="drawing" width="800"/>
    <img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/source_multi_domain/figures/age_gallery.png" alt="drawing" width="800"/>
</p>

## Train and Inference
- After setting up the dataset path, you can run the training code as shown below:

```
    python train_architecture.py
```
- To run the testing code, which will save scores in csv file.
```
    python test_architecture.py
```

## Pre-trained model
The pre-trained model can be found in [link](https://drive.google.com/drive/folders/1CHIzOUyy3YvpDi-gP6nCIdOPHJWWxQQo?usp=sharing), or you can find in the `source/save_model_trained` folder.

## Reference
If you would like to use our work, please cite:
```Bibtex
@inproceedings{xiaoguo2022MDFAS,
    title={Multi-domain Learning for Updating Face Anti-spoofing Models},
    author={Guo, Xiao and Liu, Yaojie and Jain, Anil and Liu, Xiaoming},
    booktitle={ECCV},
    year={2022}
}
```
This github will continue to update in the near future. If you have any question, please contact: [Xiao Guo](guoxia11@msu.edu) 
