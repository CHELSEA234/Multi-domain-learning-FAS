import cv2
import tensorflow as tf
import glob
import random
RANDOM_SEED = 123456789
import numpy as np
from natsort import natsorted, ns
from utils import face_crop_and_resize 
from warp import generate_offset_map, generate_uv_map
# autotune = tf.data.experimental.AUTOTUNE
autotune = -1
uv=[[0.19029412,0.19795537 ,0.21318457 ,0.22828290 ,0.24970947 ,0.28816611 ,0.33394283 ,0.39239809 ,0.47876307 ,0.56515092 ,0.62323409 ,0.66867208 ,0.70676976 ,0.72820741 ,0.74272829 ,0.75663871 ,0.76398379 ,0.25338903 ,0.28589997 ,0.32738855 ,0.36722445 ,0.40321609 ,0.55088127 ,0.58705842 ,0.62712812 ,0.66933709 ,0.70184904 ,0.47813031 ,0.47830373 ,0.47872066 ,0.47870359 ,0.43102017 ,0.45095450 ,0.47804111 ,0.50489837 ,0.52461874 ,0.30827355 ,0.33330417 ,0.36890128 ,0.40203944 ,0.37214473 ,0.33496466 ,0.55122417 ,0.58458656 ,0.62106317 ,0.64688802 ,0.61956245 ,0.58191341 ,0.37796655 ,0.41338006 ,0.45562238 ,0.47811818 ,0.50052267 ,0.54254669 ,0.57570505 ,0.54044306 ,0.51024377 ,0.47821599 ,0.44642609 ,0.41657540 ,0.38790068 ,0.44901687 ,0.47766650 ,0.50653827 ,0.56918079 ,0.50583494 ,0.47757983 ,0.44971457],
    [0.55190903,0.47428983 ,0.40360034 ,0.33980367 ,0.27118790 ,0.21624640 ,0.18327993 ,0.15577883 ,0.14014046 ,0.15676366 ,0.18313733 ,0.21531384 ,0.26951864 ,0.33780637 ,0.40212137 ,0.47324431 ,0.55168754 ,0.63735390 ,0.66241443 ,0.67068136 ,0.66713846 ,0.65712863 ,0.65805173 ,0.66828096 ,0.67205220 ,0.66368717 ,0.63796753 ,0.58252430 ,0.53523010 ,0.48812559 ,0.44775373 ,0.41256407 ,0.40846801 ,0.40317070 ,0.40854913 ,0.41281027 ,0.58095986 ,0.59604895 ,0.59652811 ,0.57966459 ,0.57139677 ,0.56953919 ,0.57967824 ,0.59695679 ,0.59599525 ,0.58050835 ,0.57008123 ,0.57134289 ,0.31730300 ,0.34064898 ,0.35593933 ,0.35154018 ,0.35593045 ,0.34062389 ,0.31715956 ,0.30086508 ,0.28950119 ,0.28752795 ,0.28963783 ,0.30076182 ,0.31932616 ,0.32959232 ,0.33032984 ,0.32936266 ,0.31900606 ,0.32014942 ,0.31873652 ,0.32043788],
    [0.54887491,0.55835652 ,0.56531715 ,0.58029217 ,0.61638439 ,0.68007606 ,0.75769442 ,0.82921398 ,0.85709274 ,0.82894272 ,0.75751764 ,0.68032110 ,0.61664295 ,0.58068472 ,0.56520522 ,0.55785143 ,0.54947090 ,0.79504120 ,0.84203368 ,0.87477297 ,0.89484525 ,0.90437353 ,0.90412331 ,0.89423305 ,0.87385195 ,0.84139013 ,0.79445726 ,0.91648984 ,0.95176858 ,0.98838627 ,0.99706292 ,0.91018295 ,0.92791700 ,0.93613458 ,0.92778808 ,0.90999144 ,0.82165444 ,0.85368645 ,0.85440493 ,0.84463143 ,0.85324180 ,0.84432119 ,0.84337026 ,0.85280263 ,0.85272932 ,0.82140154 ,0.84402239 ,0.85248041 ,0.86857969 ,0.91266698 ,0.93638903 ,0.93873996 ,0.93629760 ,0.91227442 ,0.86774820 ,0.90530455 ,0.92216164 ,0.92610627 ,0.92281538 ,0.90596151 ,0.87151438 ,0.91635096 ,0.92336667 ,0.91626322 ,0.87006092 ,0.91713434 ,0.92056626 ,0.91682398]]
uv = np.transpose(np.asarray(uv, dtype=np.float32))
lm_ref = [[42.022587,44.278061,48.761536,53.206482,59.514465,70.836105,84.312767,101.52200,126.94785,152.38043,169.48012,182.85706,194.07301,200.38426,204.65921,208.75444,210.91682,60.597733,70.168953,82.383194,94.110878,104.70682,148.17944,158.83000,170.62653,183.05284,192.62436,126.76157,126.81262,126.93536,126.93034,112.89234,118.76100,126.73531,134.64207,140.44775,76.755737,84.124748,94.604538,104.36041,95.559410,84.613594,148.28040,158.10228,168.84100,176.44383,168.39919,157.31531,97.273354,107.69909,120.13522,126.75800,133.35388,145.72574,155.48756,145.10645,136.21576,126.78679,117.42784,108.63980,100.19796,118.19057,126.62502,135.12486,153.56682,134.91780,126.59950,118.39597],
          [94.517975,117.36908,138.18005,156.96179,177.16229,193.33707,203.04239,211.13872,215.74265,210.84879,203.08437,193.61160,177.65372,157.54980,138.61548,117.67688,94.583191,69.363007,61.985199,59.551407,60.594437,63.541336,63.269577,60.258087,59.147827,61.610504,69.182358,85.504852,99.428253,113.29582,125.18130,135.54114,136.74701,138.30655,136.72314,135.46866,85.965424,81.523193,81.382126,86.346741,88.780792,89.327667,86.342728,81.255920,81.539001,86.098343,89.168091,88.796661,163.58600,156.71295,152.21146,153.50656,152.21408,156.72034,163.62823,168.42532,171.77084,172.35178,171.73062,168.45572,162.99039,159.96802,159.75090,160.03563,163.08463,162.74802,163.16397,162.66309]]
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
        # n_stype = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        # dmap = np.concatenate([np.zeros_like(dmap0), np.ones_like(dmap0)], axis=2)
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
        # elif mode == 'test_A':
        elif 'test_A' in mode:
            data_dir_li = self.config.LI_DATA_DIR_TEST
            data_dir_sp = self.config.SP_DATA_DIR_TEST
        # elif mode == 'test_B':
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

    def inputs(self, data_dir_li, data_dir_sp):
        mode =  self.mode
        protocol = self.config.SET

        if mode == 'train' or mode == 'val':
            if self.config.dataset == 'oulu':
                li_data_samples = data_dir_li
                sp_data_samples = data_dir_sp
            else:
                li_data_samples = []
                for _dir in data_dir_li:
                    # age is different from the spoof-attack type.
                    _list = glob.glob(_dir)
                    li_data_samples += _list

                sp_data_samples = []
                for _dir in data_dir_sp:
                    _list = glob.glob(_dir)
                    sp_data_samples += _list

            data_samples = [li_data_samples, sp_data_samples]
            li_data_samples = 3500 * li_data_samples
            sp_data_samples = 2000 * sp_data_samples
            if mode =='train':
                li_data_samples = li_data_samples[:20000]
                sp_data_samples = sp_data_samples[:20000]
            else: # GX: 500 is enough for the Live_lite and Spoof_lite.
                li_data_samples = li_data_samples[:500]
                sp_data_samples = sp_data_samples[:500]

            # print("==================================")
            # print("constructing the data_samples list")
            # print(li_data_samples[:10])
            # print(sp_data_samples[:10])
            # print("==================================")

            shuffle_buffer_size = min(len(li_data_samples), len(sp_data_samples))
            dataset = tf.data.Dataset.from_tensor_slices((li_data_samples, sp_data_samples))
            dataset = dataset.shuffle(shuffle_buffer_size).repeat(-1)
            if mode == 'train':
                dataset = dataset.map(map_func=self.parse_fn, num_parallel_calls=autotune)
            elif mode == 'val':
                dataset = dataset.map(map_func=self.parse_fn_val, num_parallel_calls=autotune)
            dataset = dataset.batch(batch_size=self.config.BATCH_SIZE).prefetch(buffer_size=autotune)
        # elif mode == 'test':
        # elif mode == 'test_A' or mode == 'test_B':
        elif 'test_A' in mode or 'test_B' in mode:
            li_data_samples = []
            for _dir in data_dir_li:
                _list = glob.glob(_dir)
                li_data_samples += _list

            sp_data_samples = []
            for _dir in data_dir_sp:
                _list = glob.glob(_dir)
                sp_data_samples += _list
            if 'csv' not in mode:
                if len(li_data_samples) >= 50:
                    random.seed(RANDOM_SEED)
                    li_data_samples = random.sample(li_data_samples, 50)
                if len(sp_data_samples) >= 50:
                    random.seed(RANDOM_SEED)
                    sp_data_samples = random.sample(sp_data_samples, 50)
            data_folders = li_data_samples + sp_data_samples
            self.data_folders = data_folders
            def list_extend(vd_list):
                new_list = []
                for idx, _file in enumerate(vd_list):
                    # if idx % 3 != 0:    # this is for the lite version.
                    #     continue
                    meta = glob.glob(_file+'/*.png')
                    meta.sort()
                    random.seed(RANDOM_SEED)
                    if 'csv' not in mode:
                        meta = random.sample(meta, 3)
                    else:
                        if self.config.dataset == 'SiWM-v2':
                            if 'test_B' in mode and self.config.type == 'illu':
                                if len(meta) > 40:
                                    meta = random.sample(meta, 40)
                                else:
                                    meta = meta
                            else:
                                meta = random.sample(meta, 20)
                        else:
                            if self.config.type != 'illu':
                                meta = random.sample(meta, 5)
                            else:
                                meta = random.sample(meta, 1)
                    # meta = random.sample(meta, 3)
                    # meta = random.sample(meta, 20)
                    new_list += meta
                    # if idx == 400:  # GX: debug here
                        # break
                return new_list

            data_samples = list_extend(data_folders)
            # print("the new list are: ")
            # print(data_samples[:10])
            # import sys;sys.exit(0)
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
            # print
        except:
            print(file_name)
            print(meta)
            import sys;sys.exit(0)
        lm_name = im_name[:-3] + 'npy'
        parts = file_name.split('/')
        # dataset = parts[-5]print(parts)print(parts)
        dataset = self.config.dataset
        if dataset == 'SiWM-v2':
            stype = parts[-1].split('_')[:-1]
            stype = '_'.join(stype)
        elif dataset == 'SiW':
            sub_id = parts[-1]
            # print("the sub id is: ", sub_id)
            spoof_id = int(sub_id.split("-")[2])
            if spoof_id == 1:
                stype = 'Live'
            elif spoof_id == 2:
                stype = 'Paper'
            # elif spoof_id == 3:
            else:   # ask Yaojie
                stype = 'Replay'
            # else:
                # assert False, print("The spoof type should be valid in SiW.", spoof_id, sub_id)
        elif dataset == 'oulu':
            vid_id = parts[-1]
            # device_id, bg_id, sub_id, spoof_id
            spoof_id = int(vid_id.split('_')[-1])
            if spoof_id == 1:
                stype = "Live"
            elif spoof_id in [2,3]:
                stype = 'Paper'
            elif spoof_id in [4,5]:
                stype = 'Replay'
            else:
                assert False,print("Please offer a valid oulu img.")
            # print(parts, stype)
            # print("...coming here...")
            # import sys;sys.exit(0)
        else:
            assert False, print("Please offer the valid dataset...")
        return im_name, lm_name, dataset, stype

    def _img_preprocess(self, file_name, dataset=None):
        while True:
            im_name, lm_name, dataset_, stype = self._img_parse(file_name)
            # if not dataset and self.config.dataset=='SiWM-v2':
            #     dataset = dataset_
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
        # reg = generate_offset_map(lm2, lm1, self.config.IMG_SIZE)
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

    # def parse_fn(self, _img1, _img2, _dmap1, _dmap2, _stype1, _stype2, _reg):
    def parse_fn(self, file1, file2):
        config = self.config
        _img1, _img2, _dmap1, _dmap2, _stype1, _stype2, _reg = self.parse_fn_val(file1, file2)
        # _img1   = tf.ensure_shape(_img1, [config.IMG_SIZE, config.IMG_SIZE, 3])
        # _img2   = tf.ensure_shape(_img2, [config.IMG_SIZE, config.IMG_SIZE, 3])
        # Data augmentation.
        _img1a = tf.image.random_contrast(_img1, 0.9, 1.1)+ tf.random.uniform([1, 1, 3], minval=-0.03, maxval=0.03)
        _img1a = tf.cond(tf.greater(tf.random.uniform([1], 0, 1)[0], 0.5),lambda: _img1a, lambda: _img1)
        _img2a = tf.image.random_contrast(_img2, 0.9, 1.1)+ tf.random.uniform([1, 1, 3], minval=-0.03, maxval=0.03)
        _img2a = tf.cond(tf.greater(tf.random.uniform([1],0,1)[0],0.5),lambda: _img2a, lambda: _img2)
        # _dmap1  = tf.ensure_shape(_dmap1,[config.IMG_SIZE, config.IMG_SIZE, 2])
        # _dmap2  = tf.ensure_shape(_dmap2,[config.IMG_SIZE, config.IMG_SIZE, 2])
        # _stype1 = tf.ensure_shape(_stype1, [15])
        # _stype2 = tf.ensure_shape(_stype2, [15])
        # _reg  = tf.ensure_shape(_reg,[config.IMG_SIZE, config.IMG_SIZE, 3])

        return _img1a, _img2a, _dmap1, _dmap2, _stype1, _stype2, _reg

    def parse_fn_test(self, file):
        config = self.config
        def _parse_function(_file):
            _file = _file.decode('UTF-8')
            ## GX: in train and val; image is chosen randomly from the list.
            # meta1 = glob.glob(_file1 + '/*.png')
            # im_name1 = meta1[random.randint(0, len(meta1) - 1)]
            ## GX: here the image is specified by the image name.
            im_name = _file
            lm_name = im_name[:-3] + 'npy'
            # dataset = 'SiWM-v2'
            dataset = config.dataset

            img = cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB) / 255.
            lm  = np.load(lm_name)
            img, lm = face_crop_and_resize(img, lm, config.IMG_SIZE, aug=False)

            return img.astype(np.float32), im_name

        image, im_name = tf.numpy_function(_parse_function, [file], [tf.float32, tf.string])
        # image, im_name = tf.numpy_function(_parse_function, [file], [tf.float32])

        image = tf.ensure_shape(image, [config.IMG_SIZE, config.IMG_SIZE, 3])
        # image   = tf.ensure_shape(image, [config.IMAGE_SIZE, config.IMAGE_SIZE, 3])
        return image, im_name
