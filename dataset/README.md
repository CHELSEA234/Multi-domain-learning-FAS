## FASMD Dataset
The FASMD Dataset is constructed on three exsiting datasets: SiW-Mv2, SiW, and Oulu-NPU. FASMD consists of five sub-datasets: dataset A is the
source domain dataset, and B, C, D and E are four target domain datasets. The details can be found in [[PDF]](http://cvlab.cse.msu.edu/pdfs/guo_liu_jain_liu_eccv2022.pdf).

<p align="center">
<img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/figures/Dataset_demo.png" alt="drawing" width="800"/>
<img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/figures/age_gallery.png" alt="drawing" width="900"/>
<img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/figures/distribution.png" alt="drawing" width="800"/>
</p>

## Usage
- Please first follow links to download OULU, SIW and SIW-Mv2 datasets.
    - OULU: [download link](https://sites.google.com/site/oulunpudatabase/)
    - SIW: [download link](http://cvlab.cse.msu.edu/siw-spoof-in-the-wild-database.html)
    - SIWM-v2: [download link](https://arxiv.org/pdf/1904.02860.pdf)
- Then config.py and data partitioning files in this page can be used to construct the dataset.

## Acknowledge
- We have used fantastic following script for age estimation and lighting estimation.
    - Age Esimation: [code link](https://github.com/yu4u/age-gender-estimation)
    - Illumination Estimation: [code link](https://github.com/zhhoper/DPR)

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
This github will continue to update in the near future. If you have any question, please contact: [Xiao Guo](guoxia11@msu.edu) 
