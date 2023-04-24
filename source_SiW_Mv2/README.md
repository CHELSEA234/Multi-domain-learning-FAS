# Multi-domain Learning for Updating Face Anti-spoofing Models

<p align="center">
<img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/source_SiW_Mv2/figures/dataset_gallery.png" alt="drawing" width="1000"/>
</p>

This project page contains **S**poof **i**n **W**ild with **M**ultiple Attacks **V**ersion 2 (SiW-Mv2) dataset. [[Arxiv]](https://arxiv.org/pdf/2208.11148.pdf) [[SiW-Mv2 Dataset]](http://cvlab.cse.msu.edu/pdfs/guo_liu_jain_liu_eccv2022_supp.pdf) 

**Our algorithm has been officially accepted and delivered to the [IAPRA ODIN](https://www.iarpa.gov/research-programs/odin) program**!

Authors: [Xiao Guo](https://scholar.google.com/citations?user=Gkc-lAEAAAAJ&hl=en), [Yaojie Liu](https://yaojieliu.github.io/), [Anil Jain](https://www.cse.msu.edu/~jain/), [Xiaoming Liu](http://cvlab.cse.msu.edu/)

> Introduction: **SiW-Mv2 Dataset** is a large-scale face anti-spoofing dataset that includes $14$ spoof attack types, and these spoof attack types are designated and verified by the IARPA ODIN program. In addition, **ALL** live subjects in SiW-Mv2 dataset participate in person during the dataset collection, and they have signed the consent form which ensures the dataset usage for the research purpose. The more details are can be found in [dataset](https://github.com/CHELSEA234/Multi-domain-learning-FAS/tree/main/source_SiW_Mv2).  

### 1. Protocols:
- Protocol I: *Known Spoof Attack Detection*. We divide live subjects and subjects of each spoof pattern into train and test splits. We train the model on the training split and report the overall performance on the test split.

- Protocol II: *Unknown Spoof Attack Detection*. We follow the leave-one-out paradigm — keep $13$ spoof attack and $80$% live subjects as the train split, and use the remaining one spoof attacks and left $20$% live subjects as the test split. We report the test split performance for both individual spoof attacks, as well as the averaged performance with standard deviation.

- Protocol III: *Cross-domain Spoof Detection*. We partition the SiW-Mv2 into $5$ sub-datasets, where each sub-dataset represents different spoof type, different age and race, as well as different illuminations. We train the model on the source domain dataset, and evaluate the model on test splits of $5$ different domains. Each sub-dataset performance, and averaged performance with standard deviation are reported

### 2. Baseline Performance

We implement SRENet as the baseline model, and evaluate this SRENet on three SiW-Mv2 protocols. Please find the details in [[paper]](http://cvlab.cse.msu.edu/pdfs/guo_liu_jain_liu_eccv2022_supp.pdf).

<p align="center">
<img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/source_SiW_Mv2/figures/baseline_performance.png" alt="drawing" width="600"/>
</p>

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
