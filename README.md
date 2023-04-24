# Multi-domain Learning for Updating Face Anti-spoofing Models

<p align="center">
<img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/source_SiW_Mv2/dataset_gallery.png" alt="drawing" width="1000"/>
</p>

This project page contains **S**poof **i**n **W**ild with **M**ultiple Attacks **V**ersion 2 (SiW-Mv2) dataset and the official implementation of our ECCV2022 oral paper "Multi-domain Learning for Updating Face Anti-spoofing Models". [[Arxiv]](https://arxiv.org/pdf/2208.11148.pdf) [[SiW-Mv2 Dataset]](http://cvlab.cse.msu.edu/pdfs/guo_liu_jain_liu_eccv2022_supp.pdf) 

**Our algorithm has been officially accepted and delivered to the [IAPRA ODIN](https://www.iarpa.gov/research-programs/odin) program**!

Authors: [Xiao Guo](https://scholar.google.com/citations?user=Gkc-lAEAAAAJ&hl=en), [Yaojie Liu](https://yaojieliu.github.io/), [Anil Jain](https://www.cse.msu.edu/~jain/), [Xiaoming Liu](http://cvlab.cse.msu.edu/)

> Introduction: **SiW-Mv2 Dataset** is a large-scale face anti-spoofing dataset that includes $14$ spoof attack types, and these spoof attack types are designated and verified by the IARPA ODIN program. In addition, **ALL** live subjects in SiW-Mv2 dataset participate in person during the dataset collection, and they have signed the consent form which ensures the dataset usage for the research purpose. The more details are can be found in [dataset](https://github.com/CHELSEA234/Multi-domain-learning-FAS/tree/main/source_SiW_Mv2).  

### 1. Setup the environment.

The main dependencies are:
  ```
    python 3.8.X
    tensorflow-gpu 2.9.1
    numpy 1.23.3
    opencv-python 4.5.2.54
  ```
The enviornment setup file is `environment.yml`, please create your own environment by:
  ```
  conda env create -f environment.yml
  ```

### 2. Quick Usage
- The pre-trained weights for $3$ different protocols and corresponding `.csv` result files can be found in this [page](https://drive.google.com/drive/folders/106TrDEeH-OOfPP4cWketphMJGXtE9sgW?usp=sharing).

- To reproduce the numerical results of the baseline, please run the following command. Result will output to the screen.
```bash 
bash csv_parser.sh
Compute the protocol I scores.
AP:  ['2.3', '2.3', '0.4', '2.3', '0.0', '7.3', '5.4', '0.0', '10.7', '0.0', ...
...
```

- For inference on a single image or a directory of images, please run the following command. Of course, users can play around with their own images.

- Results will output to the screen and saved into the `.csv` file.
```bash 
bash inference.sh
...
- Results written to ./result/result.csv
...
./demo/1.png is classified as Spoof with the score 0.52
```

### 3. Train and Testing
<p align="center">
    <img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/source_SiW_Mv2/train_tb.png" alt="drawing" width="500"/>
    <img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/source_SiW_Mv2/intermediate_result.png" alt="drawing" width="300"/>
</p>

- We provide detailed dataset preprocessing steps as well as the training scripts. After following our instructions, user can generate tensorboard similar to the left figure above, and the intermediate results (right figure above) which has, from the top to down, original input image, pseudo reconstructed live images, spoof trace, ground truth and predicted depth maps. 

#### 3.1. Data Preparation
- Please first sign the [DRA form](https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/source_SiW_Mv2/DRA_form_SIWMv2.pdf) before donwloading the SiW-Mv2 dataset. 
- To preprocess the videos for training and testing, you can adapt our `preprocessing.py` for your own data configuration, such as:

```bash
python preprocessing.py
```

- After the preprocessing step completes, the program outputs extracted frames (`*.png`) and facial landmarks (`*.npy`) to `data` folder. Specifically,

```bash
./preprocessed_image_train
    ├── Train
    │     │── live
    │     │     │── Live_0
    │     │     │     │── 1.png
    │     │     │     │── 1.npy
    │     │     │     │── 2.png
    │     │     │     │── 2.npy
    │     │     │     └── ...
    │     │     │── Live_1
    │     │     │     │── 1.png
    │     │     │     │── 1.npy
    │     │     │     └── ...
    │     │     ...
    │     └── spoof
    │           │── Spoof_0
    │           │     │── 1.png
    │           │     │── 1.npy
    │           │     └── ...
    │           │── Spoof_1
    │           │     │── 1.png
    │           │     │── 1.npy
    │           │     └── ...
    │           ...
    └── Test (consistent with the training dataset configuration)
          │── live
          │     │── Live_0
          │     │     └── ...
          │     ...
          └── spoof
                │── Spoof_0
                │     └── ...
                ...
```

### 3.2. Train and Testing
- After setting up the dataset path, you can run the training code as shown below:

```
    python train_architecture.py --pro=1 --cuda=0
```
- To run the testing code, which will save scores in csv file.
```
    python test_architecture.py --pro=1 --cuda=0
```
- To run the algorithm for all $3$ protocols, please run the following code.
```
    bash run.sh
```

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
