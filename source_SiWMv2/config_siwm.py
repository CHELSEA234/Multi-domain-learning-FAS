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
import os
import abc
import csv
from glob import glob
from utils import file_reader

# Configuration class.
class Config(object):
	"""
	the meta configuration class.

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
	# Config.
	LOG_DEVICE_PLACEMENT = False
	IMG_SIZE = 256
	MAP_SIZE = 32
	FIG_SIZE = 128	

	# Training meta.
	STEPS_PER_EPOCH = 1000
	IMG_LOG_FR = 100
	TXT_LOG_FR = 1000
	
	# Initial learning rate.
	lr = 1e-4	
	LEARNING_RATE_DECAY_FACTOR = 0.89  	# The decay to use for the moving average.
	LEARNING_MOMENTUM = 0.999   
	MOVING_AVERAGE_DECAY = 0.9999		# The decay to use for the moving average.       
	GAN = 'ls' # 'hinge', 'ls'
	DECAY_STEP = 3
	n_layer_D = 4

	# Spoof type dictionary.
	spoof_type_dict = {'Co': 'Makeup_Co', 'Im': 'Makeup_Im', 'Ob': 'Makeup_Ob',
					   'Half': 'Mask_Half', 'Mann': 'Mask_Mann', 'Paper': 'Mask_Paper',
					   'Sil': 'Mask_Silicone', 'Trans': 'Mask_Trans', 'Print': 'Paper',
					   'Eye': 'Partial_Eye', 'Funnyeye': 'Partial_Funnyeye',
					   'Mouth': 'Partial_Mouth', 'Paperglass': 'Partial_Paperglass',
					   'Replay': 'Replay'}

	def __init__(self, args):
		self.MAX_EPOCH = args.epoch
		self.GPU_INDEX = args.cuda
		self.phase = args.stage
		assert self.phase in ['pretrain', 'ft', 'ub'], print("Please offer the valid phase!")
		self.type  = args.type
		self.SET = args.set
		self.illu_dict = dict()
		self.spoof_type_list = list(self.spoof_type_dict.keys())
		gpus = tf.config.experimental.list_physical_devices('GPU')
		if gpus:
			try:
				tf.config.experimental.set_memory_growth(gpus[self.GPU_INDEX], True)
				tf.config.experimental.set_visible_devices(gpus[self.GPU_INDEX], 'GPU')
				logical_gpus = tf.config.experimental.list_logical_devices('GPU')
				print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
			except RuntimeError as e:
				print(e) # Virtual devices must be set before GPUs have been initialized

	@abc.abstractmethod
	def search_folder(self, root_dir, sub_id, stype):
		pass

	@abc.abstractmethod
	def search_folder_wrapper(self, root_dir, filenames):
		pass

class Config_siwm(Config):
	"""
	the configuration class for siw-mv2 dataset.
	"""
	LI_DATA_DIR = []
	SP_DATA_DIR = []
	LI_DATA_DIR_TEST = []
	SP_DATA_DIR_TEST = []

	def __init__(self, args):
		super().__init__(args)
		self.dataset = "SiWM-v2"
		self.BATCH_SIZE = 4
		self.spoof_img_root = 'Spoof Image Directory'
		self.live_img_root  = 'Live Image Directory'
		self.protocol = args.pro
		if self.protocol == 1:
			self.unknown = 'None'
		elif self.protocol == 2:
			self.unknown  = args.unknown
			assert self.unknown in self.spoof_type_list, print("Please offer a valid spoof type.")

		root_dir_id = "Your Protocol Directory"
		self.spoof_train_fname = file_reader(root_dir_id + 'trainlist_all.txt')
		self.spoof_test_fname  = file_reader(root_dir_id + 'testlist_all.txt')
		self.live_train_fname  = file_reader(root_dir_id + 'trainlist_live.txt')
		self.live_test_fname   = file_reader(root_dir_id + 'testlist_live.txt')

	# overriding the compile method.
	def compile(self, dataset_name='SiWM-v2'):
		'''generates train and test list for SIW-Mv2.'''
		# Train data.
		for x in self.live_train_fname:
			if x != '':
				self.LI_DATA_DIR.append(self.live_img_root + x)
		for x in self.spoof_train_fname:
			if x != '':
				if self.protocol == 1:
					self.SP_DATA_DIR.append(self.spoof_img_root + x)
				elif self.protocol == 2:
					if self.spoof_type_dict[self.unknown] not in x:
						self.SP_DATA_DIR.append(self.spoof_img_root + x)

		for x in self.live_test_fname:
			if x != '':
				self.LI_DATA_DIR_TEST.append(self.live_img_root + x)
		if self.protocol == 1:
			for x in self.spoof_test_fname:
				if x != '':
					self.SP_DATA_DIR_TEST.append(self.spoof_img_root + x)
		elif self.protocol == 2:
			self.SP_DATA_DIR_TEST = glob(self.spoof_img_root + self.spoof_type_dict[self.unknown] + '_*')