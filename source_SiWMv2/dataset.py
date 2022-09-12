# Copyright 2022
# 
# Authors: Xiao Guo, Yaojie Liu, Anil Jain, and Xiaoming Liu.
# 
# All Rights Reserved.s
# 
# This research is based upon work supported by the Office of the Director of 
# National Intelligence (ODNI), Intelligence Advanced Research Projects Activity
# (IARPA), via IARPA R&D Contract No. 2017-17020200004. The views and 
# conclusions contained herein are those of the authors and should not be 
# interpreted as necessarily representing the official policies or endorsements,
# either expressed or implied, of the ODNI, IARPA, or the U.S. Government. The 
# U.S. Government is authorized to reproduce and distribute reprints for 
# Governmental purposes not withstanding any copyright annotation thereon. 
# ==============================================================================
import cv2
import tensorflow as tf
import glob
import random
import numpy as np
from natsort import natsorted, ns
from utils import face_crop_and_resize 
from warp import generate_uv_map
from parameters import uv, lm_ref, RANDOM_SEED, REPEAT_TIME_LI, REPEAT_TIME_SP, SAMPLE_NUM_TRAIN, SAMPLE_NUM_TEST

autotune = tf.data.experimental.AUTOTUNE
autotune = -1
uv = np.transpose(np.asarray(uv, dtype=np.float32))
lm_ref = np.transpose(np.asarray(lm_ref, dtype=np.float32))/256.

def get_dmap_and_stype(config, lm, dataset, stype):
    dmap0 = generate_uv_map(lm, uv, config.IMG_SIZE)
    dmap_up = np.copy(dmap0)
    dmap_up[config.IMG_SIZE//2:,:,:]=0
    dmap_bot = np.copy(dmap0)
    dmap_bot[:config.IMG_SIZE//2,:,:]=0
    if stype == 'Live':
        n_stype = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        dmap = np.concatenate([dmap0, dmap0*0], axis=2)
    elif stype == 'Makeup_Co':
        n_stype = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        dmap = np.concatenate([np.zeros_like(dmap0), dmap0], axis=2)
    elif stype == 'Makeup_Im':
        n_stype = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
        dmap = np.concatenate([np.zeros_like(dmap0), dmap0], axis=2)
    elif stype == 'Makeup_Ob':
        n_stype = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
        dmap = np.concatenate([np.zeros_like(dmap0), dmap0], axis=2)
    elif stype == 'Mask_Half':
        n_stype = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
        dmap = np.concatenate([np.zeros_like(dmap0), dmap0], axis=2)
    elif stype == 'Mask_Silicone':
        n_stype = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
        dmap = np.concatenate([np.zeros_like(dmap0), dmap0], axis=2)
    elif stype == 'Mask_Trans':
        n_stype = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
        dmap = np.concatenate([np.zeros_like(dmap0), dmap0], axis=2)
    elif stype == 'Mask_Paper':
        n_stype = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
        dmap = np.concatenate([np.zeros_like(dmap0), dmap0], axis=2)
    elif stype == 'Mask_Mann':
        n_stype = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
        dmap = np.concatenate([np.zeros_like(dmap0), dmap0], axis=2)
    elif stype == 'Partial_Funnyeye':
        n_stype = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
        dmap = np.concatenate([dmap_bot, dmap_up], axis=2)
    elif stype == 'Partial_Eye':
        n_stype = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
        dmap = np.concatenate([dmap_bot, dmap_up], axis=2)
    elif stype == 'Partial_Mouth':
        n_stype = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
        dmap = np.concatenate([dmap_up, dmap_bot], axis=2)
    elif stype == 'Partial_Paperglass':
        n_stype = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        dmap = np.concatenate([dmap_bot, dmap_up], axis=2)
    elif stype == 'Replay':
        n_stype = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        dmap = np.concatenate([np.zeros_like(dmap0), np.ones_like(dmap0)], axis=2)
    elif stype == 'Paper':
        n_stype = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        dmap = np.concatenate([np.zeros_like(dmap0), np.ones_like(dmap0)], axis=2)
    else:
        assert False, print(f"{stype} is invalid....")
    return dmap, n_stype

class Dataset():
    def __init__(self, config, mode, dset=None):
        self.config = config
        self.mode = mode
        self.dset = dset
        if mode == 'train':
            data_dir_li = self.config.LI_DATA_DIR
            data_dir_sp = self.config.SP_DATA_DIR
        elif mode == 'val':
            data_dir_li = self.config.LI_DATA_DIR_VAL
            data_dir_sp = self.config.SP_DATA_DIR_VAL
        elif 'test_A' in mode:
            data_dir_li = self.config.LI_DATA_DIR_TEST
            data_dir_sp = self.config.SP_DATA_DIR_TEST
        elif 'test_B' in mode:
            data_dir_li = self.config.LI_DATA_DIR_TEST_B
            data_dir_sp = self.config.SP_DATA_DIR_TEST_B
        self.data_folders = None
        self.data_samples = None
        self.input_tensors, self.name_list = self.inputs(data_dir_li, data_dir_sp)
        self.feed = iter(self.input_tensors)

    def __len__(self):
        return len(self.name_list)

    def _info(self):
        return len(self.data_samples)

    def nextit(self):
        return next(self.feed)

    def _return_list(self, dir):
        dir_list = []
        for _ in dir:
            _list = glob.glob(_)
            dir_list += _list
        return dir_list

    def _extend_list(self, vd_list):
        new_list = []
        for idx, _file in enumerate(vd_list):
            meta = glob.glob(_file+'/*.png')
            meta.sort()
            random.seed(RANDOM_SEED)
            meta = random.sample(meta, 20)
            new_list += meta
        return new_list

    def inputs(self, data_dir_li, data_dir_sp):
        mode =  self.mode
        protocol = self.config.SET
        if mode == 'train' or mode == 'val':
            li_data_samples = data_dir_li if self.config.dataset == 'oulu' else self._return_list(data_dir_li)
            sp_data_samples = data_dir_sp if self.config.dataset == 'oulu' else self._return_list(data_dir_sp)
            data_samples = [li_data_samples, sp_data_samples]
            li_data_samples = REPEAT_TIME_LI * li_data_samples
            sp_data_samples = REPEAT_TIME_SP * sp_data_samples
            li_data_samples = li_data_samples[:SAMPLE_NUM_TRAIN] if mode == 'train' else li_data_samples[:SAMPLE_NUM_TEST]
            sp_data_samples = sp_data_samples[:SAMPLE_NUM_TRAIN] if mode == 'train' else sp_data_samples[:SAMPLE_NUM_TEST]
            shuffle_buffer_size = min(len(li_data_samples), len(sp_data_samples))
            dataset = tf.data.Dataset.from_tensor_slices((li_data_samples, sp_data_samples))
            dataset = dataset.shuffle(shuffle_buffer_size).repeat(-1)
            if mode == 'train':
                dataset = dataset.map(map_func=self.parse_fn, num_parallel_calls=autotune)
            elif mode == 'val':
                dataset = dataset.map(map_func=self.parse_fn_val, num_parallel_calls=autotune)
            dataset = dataset.batch(batch_size=self.config.BATCH_SIZE).prefetch(buffer_size=autotune)
        elif 'test_A' in mode or 'test_B' in mode:
            li_data_samples = self._return_list(data_dir_li)
            sp_data_samples = self._return_list(data_dir_sp)
            if 'csv' not in mode:
                random.seed(RANDOM_SEED)
                li_data_samples = random.sample(li_data_samples, 50) if len(li_data_samples) >= 50 else \
                                  random.sample(li_data_samples, len(li_data_samples))
                random.seed(RANDOM_SEED)
                sp_data_samples = random.sample(sp_data_samples, 50) if len(sp_data_samples) >= 50 else \
                                  random.sample(sp_data_samples, len(sp_data_samples))
            self.data_folders = li_data_samples + sp_data_samples
            # self.data_folders = sp_data_samples + li_data_samples
            # print("the li dataset is: ", len(li_data_samples), li_data_samples[0])
            # print("the sp dataset is: ", len(sp_data_samples), sp_data_samples[0])
            # import sys;sys.exit(0)
            data_samples = self._extend_list(self.data_folders)
            dataset = tf.data.Dataset.from_tensor_slices(data_samples)
            dataset = dataset.cache()
            dataset = dataset.map(map_func=self.parse_fn_test) 
            dataset = dataset.batch(batch_size=self.config.BATCH_SIZE).prefetch(buffer_size=autotune)       
            self.data_samples = data_samples
        return dataset, data_samples

    def _img_parse(self, file_name):
        file_name = file_name.decode('UTF-8')
        meta = glob.glob(file_name + '/*.png')
        try:
            im_name = meta[random.randint(0, len(meta) - 1)]
        except:
            print(file_name)
            print(meta)
            import sys;sys.exit(0)
        lm_name = im_name[:-3] + 'npy'
        parts = file_name.split('/')
        dataset = self.config.dataset
        if dataset == 'SiWM-v2':
            stype = parts[-1].split('_')[:-1]
            stype = '_'.join(stype)
        elif dataset == 'SiW':
            spoof_id = int(parts[-1].split("-")[2])
            if spoof_id == 1:
                stype = 'Live'
            elif spoof_id == 2:
                stype = 'Paper'
            else:  
                stype = 'Replay'
        elif dataset == 'oulu':
            # device_id, bg_id, sub_id, spoof_id
            spoof_id = int(parts[-1].split('_')[-1]) 
            if spoof_id == 1:
                stype = "Live"
            elif spoof_id in [2,3]:
                stype = 'Paper'
            elif spoof_id in [4,5]:
                stype = 'Replay'
        else:
            assert False, print("Please offer the valid dataset...")
        return im_name, lm_name, dataset, stype

    def _img_preprocess(self, file_name, dataset=None):
        while True:
            im_name, lm_name, dataset_, stype = self._img_parse(file_name)
            dataset = self.config.dataset
            img = cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB) / 255.
            lm  = np.load(lm_name)
            img, lm = face_crop_and_resize(img, lm, self.config.IMG_SIZE, aug=True)
            try:
                dmap, n_stype = get_dmap_and_stype(self.config, lm, dataset, stype)
                n_stype = np.reshape(np.array([n_stype], np.float32), (-1))
            except:
                print(f"{file_name} cannot work on get_dmap_and_stype.")
                continue
            else:
                break
        return img, dmap, n_stype, lm, dataset

    def _parse_function(self, _file1, _file2):
        img1, dmap1, n_stype1, lm1, dataset = self._img_preprocess(_file1)
        img2, dmap2, n_stype2, lm2, _       = self._img_preprocess(_file2, dataset)
        reg = img1 # dummy code.
        return img1.astype(np.float32), img2.astype(np.float32), \
               dmap1.astype(np.float32), dmap2.astype(np.float32), \
               n_stype1.astype(np.float32), n_stype2.astype(np.float32), \
               reg.astype(np.float32)

    def parse_fn_val(self, file1, file2):
        config = self.config
        _img1, _img2, _dmap1, _dmap2, _stype1, _stype2, _reg = \
                                    tf.numpy_function(self._parse_function, 
                                                     [file1, file2], 
                                                     [tf.float32, tf.float32, tf.float32, 
                                                      tf.float32, tf.float32, tf.float32, 
                                                      tf.float32])
        _img1   = tf.ensure_shape(_img1, [config.IMG_SIZE, config.IMG_SIZE, 3])
        _img2   = tf.ensure_shape(_img2, [config.IMG_SIZE, config.IMG_SIZE, 3])
        _dmap1  = tf.ensure_shape(_dmap1,[config.IMG_SIZE, config.IMG_SIZE, 2])
        _dmap2  = tf.ensure_shape(_dmap2,[config.IMG_SIZE, config.IMG_SIZE, 2])
        _stype1 = tf.ensure_shape(_stype1, [15])
        _stype2 = tf.ensure_shape(_stype2, [15])
        _reg  = tf.ensure_shape(_reg,[config.IMG_SIZE, config.IMG_SIZE, 3])
        return _img1, _img2, _dmap1, _dmap2, _stype1, _stype2, _reg

    def parse_fn(self, file1, file2):
        config = self.config
        _img1, _img2, _dmap1, _dmap2, _stype1, _stype2, _reg = self.parse_fn_val(file1, file2)
        # Data augmentation.
        _img1a = tf.image.random_contrast(_img1, 0.9, 1.1)+ tf.random.uniform([1, 1, 3], minval=-0.03, maxval=0.03)
        _img1a = tf.cond(tf.greater(tf.random.uniform([1], 0, 1)[0], 0.5),lambda: _img1a, lambda: _img1)
        _img2a = tf.image.random_contrast(_img2, 0.9, 1.1)+ tf.random.uniform([1, 1, 3], minval=-0.03, maxval=0.03)
        _img2a = tf.cond(tf.greater(tf.random.uniform([1],0,1)[0],0.5),lambda: _img2a, lambda: _img2)
        return _img1a, _img2a, _dmap1, _dmap2, _stype1, _stype2, _reg

    def parse_fn_test(self, file):
        config = self.config
        def _parse_function(_file):
            _file = _file.decode('UTF-8')
            im_name = _file
            lm_name = im_name[:-3] + 'npy'
            dataset = config.dataset
            img = cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB) / 255.
            lm  = np.load(lm_name)
            img, lm = face_crop_and_resize(img, lm, config.IMG_SIZE, aug=False)
            return img.astype(np.float32), im_name
        image, im_name = tf.numpy_function(_parse_function, [file], [tf.float32, tf.string])
        image = tf.ensure_shape(image, [config.IMG_SIZE, config.IMG_SIZE, 3])
        return image, im_name
