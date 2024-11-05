## An Important Note:

Dear SiW-Mv2 Users:

  Although SiW-Mv2 is thoroughly described in this ECCV work, the reference for SiW-Mv2 in our publication was mistakenly listed as the work that originally introduced SiW. If you use the SiW-Mv2 dataset, **please consider citing us. We greatly appreciate your acknowledgment and understanding**.

  ```Bibtex
  @inproceedings{xiaoguo2022MDFAS,
      title={Multi-domain Learning for Updating Face Anti-spoofing Models},
      author={Guo, Xiao and Liu, Yaojie and Jain, Anil and Liu, Xiaoming},
      booktitle={ECCV},
      year={2022}
  }
  ```

Best regards,

The SiW-Mv2 Management Team

# SiW-Mv2 Dataset and Multi-domain FAS

<p align="center">
<img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/source_SiW_Mv2/figures/dataset_gallery.png" alt="drawing" width="1000"/>
</p>

This project page contains **S**poof **i**n **W**ild with **M**ultiple Attacks **V**ersion 2 (SiW-Mv2) dataset and the official implementation of our ECCV2022 oral paper "Multi-domain Learning for Updating Face Anti-spoofing Models". [[Arxiv]](https://arxiv.org/pdf/2208.11148.pdf) [[SiW-Mv2 Dataset]](http://cvlab.cse.msu.edu/pdfs/guo_liu_jain_liu_eccv2022_supp.pdf) 

Authors: [Xiao Guo](https://scholar.google.com/citations?user=Gkc-lAEAAAAJ&hl=en), [Yaojie Liu](https://yaojieliu.github.io/), [Anil Jain](https://www.cse.msu.edu/~jain/), [Xiaoming Liu](http://cvlab.cse.msu.edu/)

## Updates:

👏 **Our algorithm has been officially accepted and delivered to the [IAPRA ODIN](https://www.iarpa.gov/research-programs/odin) program**! 

🔥🔥**Check out our quick demo:**

<p float="left">
  <img src="source_SiW_Mv2/figures/demo_1.gif" width="300" height="200"/>
  <img src="source_SiW_Mv2/figures/demo_2.gif" width="300" height="200"/>
</p>
<p float="left">
  <img src="source_SiW_Mv2/figures/demo_3.gif" width="300" height="200"/>
  <img src="source_SiW_Mv2/figures/demo_4.gif" width="300" height="200"/>
</p>

The quick view on the code structure. 
```bash
./Multi-domain-learning-FAS
    ├── source_SiW_Mv2 (The spoof detection baseline source code, pre-trained weights and protocol partition files,.)
    ├── source_multi_domain (The multi-domain updating source code)
    └── DRA_form_SIWMv2.pdf (Dataset Release Agreement)
```
Note that the spoof detection baseline is described in the supplementary section of [[Arxiv](https://arxiv.org/pdf/2208.11148.pdf).]

## 1. SiW-Mv2 Introduction:
> Introduction: **SiW-Mv2 Dataset** is a large-scale face anti-spoofing (FAS) dataset that is first introduced in the multi-domain FAS updating algorithm. The SiW-Mv2 dataset includes 14 spoof attack types, and these spoof attack types are designated and verified by the IARPA ODIN program. Also, SiW-Mv2 dataset is a *privacy-aware* dataset, in which ALL live subjects in SiW-Mv2 dataset have signed the consent form which ensures the dataset usage for the research purpose.  The more details are can be found in [page](https://github.com/CHELSEA234/Multi-domain-learning-FAS/tree/main/source_SiW_Mv2) and [[paper]](http://cvlab.cse.msu.edu/pdfs/guo_liu_jain_liu_eccv2022_supp.pdf).  

## 2. SiW-Mv2 Protocols:
To set a baseline for future study on SiW-Mv2, we define three protocols. Note the partition file for each protocol is fixed, which can be found in `./source_SiW_Mv2/pro_3_text/` of [Dataset Sec.1](https://github.com/CHELSEA234/Multi-domain-learning-FAS/tree/main/source_SiW_Mv2#1-setup-the-environment).

- Protocol I: *Known Spoof Attack Detection*. We divide live subjects and subjects of each spoof pattern into train and test splits. We train the model on the training split and report the overall performance on the test split.

- Protocol II: *Unknown Spoof Attack Detection*. We follow the leave-one-out paradigm — keep $13$ spoof attack and $80$% live subjects as the train split, and use the remaining one spoof attacks and left $20$% live subjects as the test split. We report the test split performance for both individual spoof attacks, as well as the averaged performance with standard deviation.

- Protocol III: *Cross-domain Spoof Detection*. We partition the SiW-Mv2 into $5$ sub-datasets, where each sub-dataset represents novel spoof type, different age and ethnicity distribution, as well as new illuminations. We train the model on the source domain dataset, and evaluate the model on test splits of $5$ different domains. Each sub-dataset performance, and averaged performance with standard deviation are reported

## 3. Baseline Performance

- We implement SRENet as the baseline model, and evaluate this SRENet on three SiW-Mv2 protocols. Please find the details in [[paper]](http://cvlab.cse.msu.edu/pdfs/guo_liu_jain_liu_eccv2022_supp.pdf).
- To quick reproduce the following numerical numbers with `.csv` result files, please go to [Dataset Sec.2](https://github.com/CHELSEA234/Multi-domain-learning-FAS/tree/main/source_SiW_Mv2#2-quick-usage).

<p align="center">
<img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/source_SiW_Mv2/figures/baseline_performance.png" alt="drawing" width="600"/>
</p>

- In `./source_SiW_Mv2`, we provide detailed dataset preprocessing steps as well as the training scripts.
<p align="center">
    <img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/source_SiW_Mv2/figures/train_tb.png" alt="drawing" width="500"/>
    <img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/source_SiW_Mv2/figures/intermediate_result.png" alt="drawing" width="300"/>
</p>

## 4. Baseline Pre-trained Weights

- Also, pre-trained weights for $3$ different protocols can be found in this [page](https://drive.google.com/drive/folders/106TrDEeH-OOfPP4cWketphMJGXtE9sgW?usp=sharing).

| Protocol | Unknown    | Download | Protocol | Unknown | Download | Protocol | Unknown | Download |
|:----:|:--------:|:----:|:----:|:--------:|:----:|:----:|:--------:|:----:|
|I|N/A|[link](https://drive.google.com/drive/folders/1fSoF-Xy1DajQvIdnO8LQtEi-waXr6OaW?usp=sharing)|II|Partial Eyes|[link](https://drive.google.com/drive/folders/1AS6J0aYIUNEv6wkEf_XLWlhqncxIptfi?usp=sharing)|II|Transparent|[link](https://drive.google.com/drive/folders/1S-Pm-iAtYdr2EBgl6qhvOmHKdwcdVw3s?usp=sharing)|
|II|Full Mask|[link](https://drive.google.com/drive/folders/1m2kvmlzOySLISlbuBe3izPazev-IO30J?usp=sharing)|II|Paper Mask|[link](https://drive.google.com/drive/folders/1ng5ax86y_Gvh_DYGJvScPW7bEzA7lY9e?usp=sharing)|II|Obfuscation|[link](https://drive.google.com/drive/folders/1PI_NdjzDsLelU8nyLRTrbYZrFA_X-k-p?usp=sharing)|
|II|Cosmetic|[link](https://drive.google.com/drive/folders/1ck0uDRvTFSzYJUwkMYZyu0KSv046-G6k?usp=sharing)|II|Paper glass|[link](https://drive.google.com/drive/folders/1nOvApxLV5t1IUSxboK0w4RtymHj6sMQ8?usp=sharing)|II|Print|[link](https://drive.google.com/drive/folders/1OlWB0MKjXrrx5Q6UkWVWkygjPNbZ_4ol?usp=sharing)|
|II|Impersonate|[link](https://drive.google.com/drive/folders/1Lt-_h3vqfVJ2f_vtOzr2oOKTVnyve2oz?usp=sharing)|II|Silicone|[link](https://drive.google.com/drive/folders/1bplxEU4G_qs5P9Udy3G3c12FmJC_6kkE?usp=share_link)|II|Replay|[link](https://drive.google.com/drive/folders/1Kkp5awJMvteEGe-9772ms3s3qxH_jj4N?usp=sharing)|
|II|FunnyEyes|[link](https://drive.google.com/drive/folders/1Fs4GxiUr3zMJhoUYb8jX-Raf1WST-o90?usp=sharing)|II|Partial Mouth|[link](https://drive.google.com/drive/folders/1Z-LcrLNv5g7NrgzuF4ba2g80mEpa14p0?usp=share_link)|II|Mannequin|[link](https://drive.google.com/drive/folders/1Lv3byEmeWtgJi23A5_6SC2mkhhLs8VHe?usp=sharing)|
|III|Cross Domain|[link](https://drive.google.com/drive/folders/1Nv2BePpjQgo2YD_CqxQ1Sv99UJn7esPB?usp=sharing)|

## 5. Download

1. SiW-Mv2 database is available under a license from Michigan State University for research purposes. Sign the Dataset Release Agreement [link](https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/DRA_form_SIWMv2.pdf).

2. Submit the request and your signed DRA to `guoxia11@msu.edu` with the following information:
    - Title: SiW-Mv2 Application
    - CC: Your advisor's email
    - Content Line 1: Your name, email, affiliation
    - Content Line 2: Your advisor's name, email, webpage
    - Attachment: Signed DRA

3. You will receive the download instructions upon approval of your usage of the database.
