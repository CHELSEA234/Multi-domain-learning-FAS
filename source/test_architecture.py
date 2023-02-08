# -*- coding: utf-8 -*-
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
import tensorflow as tf
import argparse
import os
import time
import math
import csv
import numpy as np
from tqdm import tqdm
from model import Generator, Discriminator, region_estimator
from utils import Logging
from dataset import Dataset
from config import Config_siwm, Config_siw, Config_oulu
from metrics import my_metrics
from tensorboardX import SummaryWriter

class SRENet(object):
	"""
	the SRENet class.

	Attributes:
	-----------
		configurations: config, config_siw, and config_oulu.
		modules: gen_pretrained, gen, RE, multi-disc and optimizers.
		various directories for checkpoints. 
		log: log handler.

	Methods:
	-----------
		basic functions: update_lr, _restore, _save.
		optimization functions: train and train_step.
	"""
	def __init__(self, config, config_siw, config_oulu):
		self.config = config
		self.config_siw  = config_siw
		self.config_oulu = config_oulu
		self.lr = config.lr
		self.bs = config.BATCH_SIZE + config_siw.BATCH_SIZE + config_oulu.BATCH_SIZE
		self.SUMMARY_WRITER = config.SUMMARY_WRITER
		
		## The modules:
		self.gen_pretrained  = Generator()
		self.RE  = region_estimator()
		self.gen = Generator(self.RE)
		self.disc1 = Discriminator(1,config.n_layer_D)
		self.disc2 = Discriminator(2,config.n_layer_D)
		self.disc3 = Discriminator(4,config.n_layer_D)
		self.gen_opt = tf.keras.optimizers.Adam(self.lr)

		# Checkpoint initialization.
		self.save_dir = config.save_model_dir
		self.checkpoint_path_g    = self.save_dir+"/gen/cp-{epoch:04d}.ckpt"
		self.checkpoint_path_re   = self.save_dir+"/ReE/cp-{epoch:04d}.ckpt"
		self.checkpoint_path_d1   = self.save_dir+"/dis1/cp-{epoch:04d}.ckpt"
		self.checkpoint_path_d2   = self.save_dir+"/dis2/cp-{epoch:04d}.ckpt"
		self.checkpoint_path_d3   = self.save_dir+"/dis3/cp-{epoch:04d}.ckpt"
		self.checkpoint_path_g_op = self.save_dir+"/g_opt/cp-{epoch:04d}.ckpt"

		self.checkpoint_dir_g    = os.path.dirname(self.checkpoint_path_g)
		self.checkpoint_dir_re   = os.path.dirname(self.checkpoint_path_re)
		self.checkpoint_dir_d1   = os.path.dirname(self.checkpoint_path_d1)
		self.checkpoint_dir_d2   = os.path.dirname(self.checkpoint_path_d2)
		self.checkpoint_dir_d3   = os.path.dirname(self.checkpoint_path_d3)
		self.checkpoint_dir_g_op = os.path.dirname(self.checkpoint_path_g_op)

		self.model_list  = [self.gen, self.RE, self.disc1, self.disc2, self.disc3]
		self.model_p_list= [self.checkpoint_path_re,
							self.checkpoint_path_g, 
							self.checkpoint_path_d1,
							self.checkpoint_path_d2,
							self.checkpoint_path_d3]
		self.model_d_list= [self.checkpoint_dir_re,
							self.checkpoint_dir_g,
							self.checkpoint_dir_d1,
							self.checkpoint_dir_d2,
							self.checkpoint_dir_d3]

		# Log class for displaying the losses.
		self.log      = Logging(config)
		self.csv_file = open(self.config.csv_file_name, mode='w')

	#############################################################################
	def inference(self, config):
		'''the main inference entrance.'''
		## setup the csv handler.
		self.csv_writer = csv.writer(self.csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		csv_title = ['Video name', 'Dataset', 'Depth', 'Region', 'Content', 'Additive', 'Label', 'test_mode']
		self.csv_writer.writerow(csv_title)

		for model_, model_dir_ in zip(self.model_list, self.model_d_list):
			epoch_suffix = f"/cp-{59:04d}.ckpt"
			current_checkpoint = model_dir_ + epoch_suffix
			model_.load_weights(current_checkpoint).expect_partial()
			print(f"loading weights for {model_dir_}.")
			start = time.time()

		## inference.
		start = time.time()
		for test_mode in ['test_A']:
			update_list = self.test_step(test_mode)
			assert len(update_list[-1]) == len(update_list[0]), print("Their length should match.")
			print ('\n*****Time for epoch {} is {} sec*****'.format(self.config.epoch_eval+1, int(time.time()-start)))
		self.csv_file.close()
		self.SUMMARY_WRITER.close()

	def test_step(self, test_mode, viz_mode=False):
		"""gathers results from three datasets."""
		result_update = []
		result_siwm = self.test_step_helper(self.config,      test_mode, viz_mode, prefix_dataset='SIWM')  
		result_siw  = self.test_step_helper(self.config_siw,  test_mode, viz_mode, prefix_dataset='SIW')
		result_oulu = self.test_step_helper(self.config_oulu, test_mode, viz_mode, prefix_dataset='Oulu')
		for i in range(len(result_siwm)):
			update_res = result_siwm[i] + result_siw[i] + result_oulu[i]
			result_update.append(update_res)
		return result_update

	def test_step_helper(self, config_cur, test_mode, viz_mode=False, prefix_dataset=None):
		tmp_bs = config_cur.BATCH_SIZE
		config_cur.BATCH_SIZE = 16 
		dataset_test = Dataset(config_cur, test_mode+'_csv')
		img_num = len(dataset_test.name_list)
		num_step = int(img_num/config_cur.BATCH_SIZE)
		d_score, p_score, c_score, n_score, label_lst = [],[],[],[],[]
		for step in tqdm(range(num_step)):
			img, img_name = dataset_test.nextit()
			img_name = img_name.numpy().tolist()
			d, p, c, n, p_area = self._test_graph(img)
			# d, p, c, n 		   = d.numpy(), p.numpy(), c.numpy(), n.numpy()
			d_score.extend(d.numpy())
			p_score.extend(p.numpy())
			c_score.extend(c.numpy())
			n_score.extend(n.numpy())
			for i in range(config_cur.BATCH_SIZE):
				img_name_cur = img_name[i].decode('UTF-8')
				if ("Live" in img_name_cur) or ("live" in img_name_cur):
					label_cur = 0
				else:
					label_cur = 1
				label_lst.append(label_cur)
				self.csv_writer.writerow([img_name_cur, prefix_dataset, d_score[i], p_score[i],
							c_score[i], n_score[i], label_cur, test_mode])
			self.log.step = step
		config_cur.BATCH_SIZE = tmp_bs
		return d_score, p_score, c_score, n_score, label_lst

	@tf.function
	def _test_graph(self, img):
		""" 
		model outputs the result.
		dmap_pred, p_area, c, n are depth, region, content, and additive traces.
		"""
		dmap_pred, p_area, c, n, x, region_map = self.gen(img, training=False)
		d = tf.reduce_mean(dmap_pred[:,:,:,0], axis=[1,2])
		p = tf.reduce_mean(p_area, axis=[1,2,3])
		c = tf.reduce_mean(c, axis=[1,2,3])
		n = tf.reduce_mean(n, axis=[1,2,3])
		return d, p, c, n, p_area

def main(args):
	# Base Configuration Class
	config, config_siw, config_oulu = Config_siwm(args), Config_siw(args), Config_oulu(args)
	config.lr   = args.lr
	config.type = args.type
	config.epoch_eval      = args.epoch_eval
	config.pretrain_folder = args.pretrain_folder
	config.desc_str = '_trained'
	config.root_dir = './log'+config.desc_str
	config.exp_dir  = '/exp'+config.desc_str
	config.CHECKPOINT_DIR = config.root_dir+config.exp_dir
	config.tb_dir   = './tb_logs'+config.desc_str
	config.csv_file_name  = config.root_dir+'/res'+config.desc_str+'.csv'
	config.save_model_dir = "./save_model"+config.desc_str
	config.SUMMARY_WRITER = SummaryWriter(config.tb_dir)
	os.makedirs(config.root_dir, exist_ok=True)
	os.makedirs(config.save_model_dir, exist_ok=True)
	os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
	os.makedirs(config.CHECKPOINT_DIR+'/test', exist_ok=True)
	print('**********************************************************')
	print(f"Making root folder: {config.root_dir}")
	print(f"Current exp saved into folder: {config.CHECKPOINT_DIR}")
	print(f"The tensorboard results are saved into: {config.tb_dir}")
	print(f"The trained weights saved into folder: {config.save_model_dir}")
	print('**********************************************************')
	config.compile(dataset_name='SiWM-v2')
	config_siw.compile(dataset_name='SiW')
	config_oulu.compile(dataset_name='Oulu')
	print('**********************************************************')

	srenet = SRENet(config, config_siw, config_oulu)
	srenet.inference(config)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--cuda', type=int, default=6, help='The gpu num to use.')
  parser.add_argument('--stage', type=str, default='ft', choices=['ft','pretrain','ub'])
  parser.add_argument('--type', type=str, default='spoof', choices=['spoof','age','race','illu'])
  parser.add_argument('--set', type=str, default='all', help='To choose from the predefined 14 types.')
  parser.add_argument('--epoch', type=int, default=60, help='How many epochs to train the model.')
  parser.add_argument('--data', type=str, default='all', choices=['all','SiW','SiWM','oulu'])
  parser.add_argument('--lr', type=float, default=1e-7, help='The starting learning rate.')
  parser.add_argument('--decay_step', type=int, default=2, help='The learning rate decay step.')
  parser.add_argument('--pretrain_folder', type=str, default='./', help='Deprecated function.')
  parser.add_argument('--epoch_eval', type=int, default=0, help='Which epoch checkpoint to evaluate.')
  args = parser.parse_args()
  main(args)
