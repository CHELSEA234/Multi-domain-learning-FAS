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
    ├── pro_3_text (text file for the three protocol)
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
