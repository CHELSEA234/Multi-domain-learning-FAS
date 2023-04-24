# -*- coding: utf-8 -*-
# Copyright 2022
# 
# Multi-domain Learning for Updating Face Anti-spoofing Models (ECCV 2022)
# Xiao Guo, Yaojie Liu, Anil Jain, and Xiaoming Liu
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
import numpy as np
import csv
from model import Generator, Discriminator, region_estimator
from utils import l1_loss, l2_loss, Logging
from dataset import Dataset
from config_siwm import Config_siwm
from tensorboardX import SummaryWriter
from tqdm import tqdm

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
	def __init__(self, config):
		self.config = config
		self.lr = config.lr
		self.bs = config.BATCH_SIZE
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
		self.csv_file = open(self.config.csv_file_name, mode='w')

	#############################################################################
	def inference(self, config):
		'''the main inference entrance.'''
		## setup the csv handler.
		self.csv_writer = csv.writer(self.csv_file, delimiter=',', quotechar='"', 
									 quoting=csv.QUOTE_MINIMAL)
		self.csv_writer.writerow(['Video name', 'Dataset', 'Depth', 'Region', \
								  'Content', 'Additive', 'Label', 'test_mode'])
		## loading the target epoch weight.
		self.log.epoch = self.config.epoch_eval
		for model_, model_dir_ in zip(self.model_list, self.model_d_list):
			current_checkpoint = model_dir_ + f"/cp-{self.config.epoch_eval:04d}.ckpt"
			model_.load_weights(current_checkpoint).expect_partial()
			print("*********************************************************")
			print(f"loading weights from {current_checkpoint}.")
			print("*********************************************************")

		## inference.
		start = time.time()
		test_mode = 'test_A'
		update_list = self.test_step(test_mode)
		assert len(update_list[-1]) == len(update_list[0]), print("Their length should match.")
		print('\n*****Time for epoch {} is {} sec*****'.format(self.config.epoch_eval+1, 
																int(time.time()-start)))
		self.csv_file.close()
		self.SUMMARY_WRITER.close()
		print("...Execution Over...")

	def test_step(self, test_mode, viz_mode=False):
		"""gathers results from three datasets."""
		result_update = []
		result_siwm = self.test_step_helper(self.config, test_mode, viz_mode, prefix_dataset='SIWM')  
		return result_siwm

	def test_step_helper(self, config_cur, test_mode, viz_mode=False, prefix_dataset=None):
		tmp_bs = config_cur.BATCH_SIZE
		config_cur.BATCH_SIZE = 64
		dataset_test = Dataset(config_cur, test_mode+'_csv')
		img_num = len(dataset_test.name_list)
		print(f"The total inference image number is: {img_num}.")
		num_step = int(img_num/config_cur.BATCH_SIZE)
		d_score, p_score, c_score, n_score, final_score, label_lst = [],[],[],[],[],[]
		for step in tqdm(range(num_step)):
			self.log.step = step
			img, img_name = dataset_test.nextit()
			img_name = img_name.numpy().tolist()
			d, p, c, n, figs = self._test_graph(img)
			d, p, c, n = d.numpy(), p.numpy(), c.numpy(), n.numpy()
			d_score.extend(d)
			p_score.extend(p)
			c_score.extend(c)
			n_score.extend(n)
			self.log.save(figs, training=False)
			for i in range(config_cur.BATCH_SIZE):
				img_name_cur = img_name[i].decode('UTF-8')
				if ("Live" in img_name_cur) or ("live" in img_name_cur):
					label_cur = 0
				else:
					label_cur = 1
				label_lst.append(label_cur)
				final_score.append(d[i]+p[i])
				self.csv_writer.writerow([img_name_cur, prefix_dataset, d[i], p[i],
										  c[i], n[i], label_cur, test_mode])
				self.csv_file.flush()
		config_cur.BATCH_SIZE = tmp_bs
		return d_score, p_score, c_score, n_score, final_score, label_lst

	@tf.function
	def _test_graph(self, img):
		""" 
		model outputs the result.
		dmap_pred, p_area, c, n are depth, region, content, and additive traces.
		"""
		dmap_pred, p_area, c, n, x, region_map = self.gen(img, training=False)
		dmap_pred = tf.concat([dmap_pred, 
								tf.zeros([dmap_pred.get_shape()[0], 32, 32, 1])], 
								axis=3)
		converted_gray = tf.image.rgb_to_grayscale(img)
		figs = [img, tf.abs(p_area), tf.abs(region_map), dmap_pred]

		d = tf.reduce_mean(dmap_pred[:,:,:,1], axis=[1,2]) - \
			tf.reduce_mean(dmap_pred[:,:,:,0], axis=[1,2])
		p = tf.reduce_mean(p_area, axis=[1,2,3])
		c = tf.reduce_mean(c, axis=[1,2,3])
		n = tf.reduce_mean(n, axis=[1,2,3])
		return d, p, c, n, figs

def main(args):
	# Base Configuration Class
	config = Config_siwm(args)
	config.lr   = args.lr
	config.type = args.type
	config.pretrain_folder = args.pretrain_folder
	config.DECAY_STEP = args.decay_step
	config.pretrain_folder = args.pretrain_folder
	config.desc_str = f'_siwmv2_pro_{args.pro}_unknown_{args.unknown}'
	config.root_dir = './log'+config.desc_str
	config.exp_dir  = '/exp'+config.desc_str
	config.CHECKPOINT_DIR = config.root_dir+config.exp_dir
	config.tb_dir   = './tb_logs'+config.desc_str
	config.save_model_dir = "./save_model"+config.desc_str
	config.csv_file_name  = config.root_dir+'/res_'+str(config.epoch_eval)+'.csv'
	config.SUMMARY_WRITER = SummaryWriter(config.tb_dir)
	print('**********************************************************')
	config.compile()
	print('**********************************************************')
	srenet = SRENet(config)
	srenet.inference(config)

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
	parser.add_argument('--epoch_eval', type=int, default=49, help='Which epoch to eval.')
	parser.add_argument('--dir', type=str, default=None, help='the inference image folder.')
	parser.add_argument('--img', type=str, default='fake.png', help='the inference image.')
	args = parser.parse_args()
	main(args)