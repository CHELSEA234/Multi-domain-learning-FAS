## SiWM-v2 Dataset Introduction:
We have curated SiW-Mv2 dataset that includes $14$ spoof types spanning from typical print and replay attack, to various masks, impersonation makeup and physical material coverings. SiW-Mv2 has *the largest variance* in terms of the spoof pattern, each of these patterns 
are designated and verified by the IARPA project. For example, obfuscation makeup and partial coverings intend to hide subject's identity; impersonate and $3$ mask modifies the subject appearance for imitating other identities. <br />
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Statistics
SiW-Mv2 collects $796$ Live videos from subjects from $493$ subjects, and $940$ Spoof videos from $597$ subjects. 
The details of each categories are in the follow images. 

<p align="center">
<img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/figures/siwmv2_dataset.png" alt="drawing" width="800"/>
</p>

## Script 
The current directory applies the proposed SRE on the SiW-Mv2 dataset. Please refer to the SRE code for the dependencies.
```bash
./source_SiWMv2
├── PROTOCOL
│      ├── trainlist_live.txt
│      ├── trainlist_all.txt
│      ├── testlist_live.txt
│      └── testlist_all.txt
│ 
├── DRA_form_SIWMv2.pdf
├── config_siwm.py
├── dataset.py
│ ...
```

### Training script: 
```
python train_architecture.py --cuda=0 --pro=3 --lr=1e-4
```


## Protocol
We propose three major protocols to evaluate the face anti-spoofing models on the known, unknown and cross-domain scenarios. 
The pre-defined protocol files are provided.
1. **Known Spoof Pattern Detection**. <br />
   We split live subjects and subjects of each spoof pattern into train and test splits. Given such training and test partition files, 
   we evaluate our model and report the performance. This protocol evaluates the model capability of detecting the known spoof pattern.
2. **Unknown Spoof Pattern Detection**. <br />
   We use the leave-one-out paradigm, which keeps $13$ spoof pattern samples in the training, and one remaining spoof pattern for the spoof section evaluation. 
   This protocol evaluates the model capability of detecting the unknown spoof pattern.
3. **Cross Domain Spoof Detection**. <br />
   We follow the attribute label (e.g., spoof pattern, race, age and illumination) in the paper, to evaluate the model performance on the unseen target domain. 
   This protocol evaluates the model generalization ability.

## Download:
SiW-Mv2 database is available under a license from Michigan State University for research purposes. <br />
Please first ask your advisor to sign the Dataset Release Agreement (DRA) form [link](https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/source_SiWMv2/DRA_form_SIWMv2.pdf). Then send this application to **guoxia11@msu.edu** for the download link.
After download and unzip the downloaded zip file, the tree structure is as following:
```bash
./source_SiWMv2
├── Live
│      ├── Train
│      └── Test
│ 
└── Spoof
       ├── Makeup_Cosmetic
       ├── Makeup_Impersonation
       ├── Makeup_Obfuscation
       ├── Mask_HalfMask
       ├── Mask_MannequinHead
       ├── Mask_PaperMask
       ├── Mask_SiliconeMask
       ├── Mask_TransparentMask
       ├── Paper
       ├── Partial_Eye
       ├── Partial_FunnyeyeGlasses
       ├── Partial_Mouth
       ├── Partial_PaperGlasses
       └── Replay
```

## Baseline performance
The baseline performance and trained weights will be updated on this page, recently.

## Questions.
Please feel free to ask questions and send it to **guoxia11@msu.edu**. All emails will be replied within less than **5** bussiness days. 

## Reference
If you would like to use our work, please cite:
```Bibtex
@inproceedings{xiaoguo2023MDFAS
      title={Multi-domain Learning for Updating Face Anti-spoofing Models}, 
      author={Xiao, Guo and Yaojie, Liu, Anil, Jain and Liu, Xiaoming},
      booktitle={In Proceeding of European Conference on Computer Vision (ECCV 2022)},
      year={2022}
      
}
@inproceedings{cvpr19yaojie,
    title={Deep Tree Learning for Zero-shot Face Anti-Spoofing},
    author={Yaojie Liu, Joel Stehouwer, Amin Jourabloo, Xiaoming Liu},
    booktitle={In Proceeding of IEEE Computer Vision and Pattern Recognition (CVPR 2019)},
    address={Long Beach, CA},
    year={2019}
}
```
This github will continue to update in the near future. If you have any question, please contact: [Xiao Guo](guoxia11@msu.edu) 
