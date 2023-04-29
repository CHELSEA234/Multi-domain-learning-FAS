# SiW-Mv2 Dataset

<p align="center">
    <img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/source_SiW_Mv2/figures/train_tb.png" alt="drawing" width="500"/>
    <img src="https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/source_SiW_Mv2/figures/intermediate_result.png" alt="drawing" width="300"/>
</p>

- We provide detailed dataset preprocessing steps as well as the training scripts. 
- After following our instructions, user can generate tensorboard similar to the left figure above, and the intermediate results (right figure above) which has, from the top to down, original input image, pseudo reconstructed live images, spoof trace, ground truth and predicted depth maps. 

### 1. Setup the environment.

- The quick view on the code structure:
```bash
./source_SiW_Mv2
    ├── config_siwm.py 
    ├── train.py
    ├── test.py
    ├── run.sh (call train.py and test.py)
    ├── inference.py
    ├── inference.sh (call inference.py for the custom data.)
    ├── csv_parser.py   
    ├── csv_parser.sh (call csv_parser.py to reproduce the numerical baseline result.)
    ├── pro_3_text (partition of the three protocol)
    │      ├── trainlist_all.txt (protocol I spoof train)
    │      ├── trainlist_live.txt (protocol I live train)
    │      ├── testlist_all.txt (protocol I spoof test)
    │      ├── testlist_live.txt (protocol I live test)
    │      ├── train_A_pretrain.txt (protocol III source domain subject)
    │      ├── train_B_spoof.txt (protocol III target domain B subject)
    │      └── ...
    ├── model.py (SRENet)
    ├── preprocessing.py (data preprocessing file.)
    ├── demo (the demo image and image dir for the quick usage)
    │      └── ...
    ├── parameters.py
    ├── enviornment.yml
    ├── metrics.py
    ├── utils.py
    ├── warp.py
    └── DRA_form_SIWMv2.pdf (Dataset Release Agreement)
```

- To create your own environment by:
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

#### 3.1. Data Preparation
- Please first sign the [DRA form](https://github.com/CHELSEA234/Multi-domain-learning-FAS/blob/main/source_SiW_Mv2/DRA_form_SIWMv2.pdf) before donwloading the SiW-Mv2 dataset. 
- After unzip the dataset files, you can obtain the following structure:
```bash
./SiW-Mv2
    ├── Spoof (contain 14 folders, each of which has raw videos).
    │     ├── Makeup_Cosmetic
    │     ├── Makeup_Impersonation
    │     ├── Makeup_Obfuscation
    │     ├── Mannequin
    │     ├── Silicone
    │     ├── Print
    │     ├── Replay
    │     ├── Partial_FunnyeyeGlasses
    │     ├── Partial_PaperGlasses
    │     ├── Partial_Eye
    │     ├── Partial_Mouth
    │     ├── Mask_HalfMask
    │     ├── Mask_PaperMask
    │     └── Mask_TransparentMask
    ├── Live (contain 785 raw video files)
    └── DRA_form_SIWMv2.pdf (Dataset Release Agreement)
```

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

#### 3.2. Train and Testing
- After setting up the dataset path, you can run the training code as shown below:
```
    python train_architecture.py --pro=1 --cuda=0
```
- Use `--pro=1` and `--unknown=Co` to decide which protocol and which unknown type is.
- `--batch_size`, `--lr`, and `--decay_step` are training hyper-parameters.
- `--cuda=0` specifies the GPU usage.

- To run the testing code, which will save scores in csv file.
```
    python test_architecture.py --pro=1 --cuda=0
```
- To run the algorithm for all $3$ protocols, please run the following code.
```
    bash run.sh
```

#### 3.3 Pre-trained Weights.
- Pre-trained weights for $3$ different protocols can be found in this [page](https://drive.google.com/drive/folders/106TrDEeH-OOfPP4cWketphMJGXtE9sgW?usp=sharing).

| Protocol | Unknown    | Download | Protocol | Unknown | Download | Protocol | Unknown | Download |
|:----:|:--------:|:----:|:----:|:--------:|:----:|:----:|:--------:|:----:|
|I|N/A|[link](https://drive.google.com/drive/folders/1fSoF-Xy1DajQvIdnO8LQtEi-waXr6OaW?usp=sharing)|II|Partial Eyes|[link](https://drive.google.com/drive/folders/1AS6J0aYIUNEv6wkEf_XLWlhqncxIptfi?usp=sharing)|II|Transparent|[link](https://drive.google.com/drive/folders/1S-Pm-iAtYdr2EBgl6qhvOmHKdwcdVw3s?usp=sharing)|
|II|Full Mask|[link](https://drive.google.com/drive/folders/1m2kvmlzOySLISlbuBe3izPazev-IO30J?usp=sharing)|II|Paper Mask|[link](https://drive.google.com/drive/folders/1ng5ax86y_Gvh_DYGJvScPW7bEzA7lY9e?usp=sharing)|II|Obfuscation|[link](https://drive.google.com/drive/folders/1PI_NdjzDsLelU8nyLRTrbYZrFA_X-k-p?usp=sharing)|
|II|Cosmetic|[link](https://drive.google.com/drive/folders/1ck0uDRvTFSzYJUwkMYZyu0KSv046-G6k?usp=sharing)|II|Paper glass|[link](https://drive.google.com/drive/folders/1nOvApxLV5t1IUSxboK0w4RtymHj6sMQ8?usp=sharing)|II|Print|[link](https://drive.google.com/drive/folders/1OlWB0MKjXrrx5Q6UkWVWkygjPNbZ_4ol?usp=sharing)|
|II|Impersonate|[link](https://drive.google.com/drive/folders/1Lt-_h3vqfVJ2f_vtOzr2oOKTVnyve2oz?usp=sharing)|II|Silicone|[link](https://drive.google.com/drive/folders/1bplxEU4G_qs5P9Udy3G3c12FmJC_6kkE?usp=share_link)|II|Replay|[link](https://drive.google.com/drive/folders/1Kkp5awJMvteEGe-9772ms3s3qxH_jj4N?usp=sharing)|
|II|FunnyEyes|[link](https://drive.google.com/drive/folders/1Fs4GxiUr3zMJhoUYb8jX-Raf1WST-o90?usp=sharing)|II|Partial Mouth|[link](https://drive.google.com/drive/folders/1Z-LcrLNv5g7NrgzuF4ba2g80mEpa14p0?usp=share_link)|II|Mannequin|[link](https://drive.google.com/drive/folders/1Lv3byEmeWtgJi23A5_6SC2mkhhLs8VHe?usp=sharing)|
|III|Cross Domain|[link](https://drive.google.com/drive/folders/1Nv2BePpjQgo2YD_CqxQ1Sv99UJn7esPB?usp=sharing)|


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
