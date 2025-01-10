# -*- coding: utf-8 -*-
# Copyright 2022
# 
# Authors: Xiao Guo.
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
from metrics import my_metrics
import argparse
import os
import csv
import time
import math
import numpy as np
from tqdm import tqdm
from glob import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    ## exp protocol setup.
    parser.add_argument('--stage', type=str, default='ft', choices=['ft','pretrain','ub'])
    parser.add_argument('--type', type=str, default='spoof', choices=['spoof','age','race','illu'])
    parser.add_argument('--set', type=str, default='all', help='To choose from the predefined 14 types.')
    parser.add_argument('--data', type=str, default='all', choices=['all','SiW','SiWM','oulu'])
    parser.add_argument('--pretrain_folder', type=str, default='./pre_trained/', help='Pretrain weight.')
    parser.add_argument('--pro', type=int, default=1, help='Protocol number.')
    parser.add_argument('--unknown', type=str, default='Ob', help='The unknown spoof type.')

    ## train hyper-parameters.
    parser.add_argument('--epoch', type=int, default=50, help='How many epochs to train the model.')
    parser.add_argument('--lr', type=float, default=1e-4, help='The starting learning rate.')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size.')
    parser.add_argument('--decay_step', type=int, default=3, help='The learning rate decay step.')
    parser.add_argument('--cuda', type=int, default=3, help='The gpu num to use.')
    parser.add_argument('--debug_mode', type=str, default='True', choices=['True', "False"], 
                        help='Deprecated function.')

    ## inference
    parser.add_argument('--weight_dir', type=str, default='.', help='pre-trained weights dirs.')
    parser.add_argument('--dir', type=str, default=None, help='the inference image folder.')
    parser.add_argument('--img', type=str, default=None, help='the inference image.')
    parser.add_argument('--warnings', action='store_true', help='show tensorflow warnings. By default, only errors are shown.')
    parser.add_argument('--overwrite', action='store_true', help='overwrite output file if already exists.')
  
    args = parser.parse_args()
  
    if not args.warnings:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        import tensorflow as tf
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

from config_siwm import Config_custom
from model import Generator, Discriminator, region_estimator
from dataset import Dataset
from utils import l1_loss, l2_loss, hinge_loss, Logging, normalization_score
from tensorboardX import SummaryWriter

class SRENet(object):
    def __init__(self, config, args):
        self.config = config
        self.lr = config.lr
        self.bs = config.BATCH_SIZE
        self.SUMMARY_WRITER = config.SUMMARY_WRITER
        if args.debug_mode == 'True':
            self.debug_mode = True
        else:
            self.debug_mode = False

        ## The new set:
        self.RE  = region_estimator()
        self.gen = Generator(self.RE)
        self.gen_pretrained = Generator(self.RE)
        self.disc1 = Discriminator(1,config.n_layer_D)
        self.disc2 = Discriminator(2,config.n_layer_D)
        self.disc3 = Discriminator(4,config.n_layer_D)
        self.gen_opt = tf.keras.optimizers.Adam(self.lr)
        self.disc_opt = tf.keras.optimizers.Adam(self.lr)

        # Checkpoint initialization.
        self.save_dir = config.save_model_dir
        self.checkpoint_path_g = self.save_dir+"/gen/cp-{epoch:04d}.ckpt"
        self.checkpoint_path_re= self.save_dir+"/ReE/cp-{epoch:04d}.ckpt"
        self.checkpoint_path_d1= self.save_dir+"/dis1/cp-{epoch:04d}.ckpt"
        self.checkpoint_path_d2= self.save_dir+"/dis2/cp-{epoch:04d}.ckpt"
        self.checkpoint_path_d3= self.save_dir+"/dis3/cp-{epoch:04d}.ckpt"
        self.checkpoint_path_g_op = self.save_dir+"/g_opt/cp-{epoch:04d}.ckpt"
        self.checkpoint_path_d_op = self.save_dir+"/d_opt/cp-{epoch:04d}.ckpt"
        
        self.checkpoint_dir_g    = os.path.dirname(self.checkpoint_path_g)
        self.checkpoint_dir_re   = os.path.dirname(self.checkpoint_path_re)
        self.checkpoint_dir_d1   = os.path.dirname(self.checkpoint_path_d1)
        self.checkpoint_dir_d2   = os.path.dirname(self.checkpoint_path_d2)
        self.checkpoint_dir_d3   = os.path.dirname(self.checkpoint_path_d3)
        self.checkpoint_dir_g_op = os.path.dirname(self.checkpoint_path_g_op)
        self.checkpoint_dir_d_op = os.path.dirname(self.checkpoint_path_d_op)

        self.model_list  = [self.gen, self.RE,
                            self.disc1, self.disc2, self.disc3]
        self.model_p_list= [self.checkpoint_path_g, 
                            self.checkpoint_path_re,
                            self.checkpoint_path_d1,
                            self.checkpoint_path_d2,
                            self.checkpoint_path_d3]
        self.model_d_list= [self.checkpoint_dir_g,
                            self.checkpoint_dir_re,
                            self.checkpoint_dir_d1,
                            self.checkpoint_dir_d2,
                            self.checkpoint_dir_d3]

        # Log class for displaying the losses.
        self.log = Logging(config)
        self.gen_opt  = tf.keras.optimizers.Adam(self.lr)
        self.disc_opt = tf.keras.optimizers.Adam(self.lr)
        self.csv_file = None
        self.txt_file = open(self.config.txt_file_name+'.txt', mode='a')
        self.pred_list = []
        self.GT_list   = []
        self.test_mode = 'inference'
        self.inference_data_dir = config.inference_data_dir
        self.inference_data_img = config.inference_data_img

    def _restore(self, model, checkpoint_dir, pretrain=False):
        last_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(last_checkpoint)
        if not pretrain:
            last_epoch = int((last_checkpoint.split('.')[1]).split('-')[-1])
            return last_epoch

    #############################################################################
    def inference(self, config):
        '''the main inference entrance.'''
        last_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir_g)
        if last_checkpoint != None:
            epoch_num = last_checkpoint.split('/')[-1].split('.')[-2]
            epoch_num = epoch_num.replace('cp-','')
            if epoch_num == '0000':
                epoch_num = 0
            else:
                epoch_num = int(epoch_num.lstrip('0'))
        else:
            epoch_num = 49 # make the 45th checkpoint public.

        for model_, model_dir_ in zip(self.model_list, self.model_d_list):
            epoch_suffix = f"/cp-{epoch_num:04d}.ckpt"
            current_checkpoint = model_dir_ + epoch_suffix
            model_.load_weights(current_checkpoint)
            start = time.time()
        print(f"loading weights at epoch {epoch_num}.")
        self.test_step(self.test_mode)

    def test_step(self, test_mode):
        if self.config.inference_data_dir != None:
            filename = self.config.csv_file_name+'.csv'
            print("overwrite: ", args.overwrite)
            if os.path.exists(filename) and not args.overwrite:
                print(f"File {filename} exists and --overwrite option was not set, aborting.")
                exit(1)
            csv_file = open(filename, mode='w')
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"',
                                     quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['Image name', 'Score', 'Decision']) 
        else:
            csv_file = None
            csv_writer = None

        old_bs = self.config.BATCH_SIZE
        self.config.BATCH_SIZE = 5
        dataset_inference = Dataset(self.config, test_mode)
        img_num = len(dataset_inference.name_list)
        num_list = int(img_num/self.config.BATCH_SIZE)+img_num%self.config.BATCH_SIZE
        final_score = None
        decision = None
        for step in tqdm(range(num_list)):
            img, img_name = dataset_inference.nextit()
            img_name = img_name.numpy().tolist()
            dmap_score, p = self._test_graph(img)
            dmap_score, p = dmap_score.numpy(), p.numpy()
            for idx, _ in enumerate(img_name):
                img_name_cur = img_name[idx].decode('UTF-8')
                img_idx = img_name_cur.split('/')[-1].replace('.png','')
                img_idx = int(img_idx)
                final_score = dmap_score[idx] + 0.1*p[idx]
                final_score, decision = normalization_score(final_score)
                if csv_writer is not None:
                    csv_writer.writerow([img_name_cur, f"{final_score:.2f}", decision])
        if self.config.inference_data_img != None:
            print(f"{img_name_cur} is classified as {decision} with the score {final_score:.2f}")
        else:
            print(f"Results written to {filename}")
        self.config.BATCH_SIZE = old_bs
        if csv_file is not None:
            csv_file.close()

    @tf.function
    def _test_graph(self, img):
        dmap_pred, p, c, n, x, region_map = self.gen(img, training=False)
        dmap_score = tf.reduce_mean(dmap_pred[:,:,:,1], axis=[1,2]) - \
                     tf.reduce_mean(dmap_pred[:,:,:,0], axis=[1,2])
        p = tf.reduce_mean(p, axis=[1,2,3])
        return dmap_score, p

def main(args):
    # Base Configuration Class
    config = Config_custom(args)
    config.lr   = args.lr
    config.type = args.type
    config.DECAY_STEP = args.decay_step
    config.pretrain_folder = args.pretrain_folder
    config.desc_str = f'_siwmv2_pro_{args.pro}_unknown_{args.unknown}'
    config.root_dir = './log'+config.desc_str
    config.exp_dir  = '/exp'+config.desc_str
    config.CHECKPOINT_DIR = config.root_dir+config.exp_dir
    config.tb_dir   = './tb_logs'+config.desc_str
    config.save_model_dir = f"{args.weight_dir}/save_model"+config.desc_str
    config.res_dir = './result'
    config.csv_file_name  = config.res_dir + '/result'
    config.txt_file_name  = config.res_dir + '/result'
    config.SUMMARY_WRITER = SummaryWriter(config.tb_dir)

    if not os.path.exists(config.res_dir):
        print('**********************************************************')
        print(f"Making results directory: {config.res_dir}")
        print('**********************************************************')
        os.makedirs(config.res_dir, exist_ok=True)
    else:
        print('**********************************************************')
        print(f"Using results directory: {config.res_dir}")
        print('**********************************************************')
    stdnet = SRENet(config, args)
    stdnet.inference(config)

if __name__ == '__main__':
    main(args)
