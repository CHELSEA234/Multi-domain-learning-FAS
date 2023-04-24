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
	spoof_type_dict = {
					'Co': 'Makeup_Co', 'Eye': 'Partial_Eye', 'Funnyeye': 'Partial_Funnyeye',
					'Half': 'Mask_Half', 'Im': 'Makeup_Im', 'Mann': 'Mask_Mann', 'Mouth': 'Partial_Mouth',
					'Ob': 'Makeup_Ob', 'Paper': 'Mask_Paper', 'Paperglass': 'Partial_Paperglass',
					'Print': 'Paper', 'Replay': 'Replay', 
					'Sil': 'Mask_Silicone', 'Trans': 'Mask_Trans'
					}

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
		self.BATCH_SIZE = args.batch_size
		self.epoch_eval = args.epoch_eval
		self.spoof_img_root = '/user/guoxia11/cvlshare/cvl-guoxia11/Spoof/'
		self.live_img_root  = '/user/guoxia11/cvlshare/cvl-guoxia11/Live/'
		self.protocol = args.pro
		if self.protocol in [1, 3]:
			self.unknown = 'None'
		elif self.protocol == 2:
			self.unknown  = args.unknown
			assert self.unknown in self.spoof_type_list, print("Please offer a valid spoof type.")

		root_dir_id = "/user/guoxia11/cvl/anti_spoofing_2022/PROTOCOL/SIW-Mv2/"
		if self.protocol in [1, 2]:
			self.spoof_train_fname = file_reader(root_dir_id + 'trainlist_all.txt')
			self.spoof_test_fname  = file_reader(root_dir_id + 'testlist_all.txt')
			self.live_train_fname  = file_reader(root_dir_id + 'trainlist_live.txt')
			self.live_test_fname   = file_reader(root_dir_id + 'testlist_live.txt')
		elif self.protocol == 3:
			total_list_train = file_reader(root_dir_id + 'train_A_pretrain.txt')
			total_list_test  = file_reader(root_dir_id + 'test_A_pretrain.txt')
			total_list_test += file_reader(root_dir_id + 'test_B_spoof.txt')
			total_list_test += file_reader(root_dir_id + 'test_C_race.txt')
			total_list_test += file_reader(root_dir_id + 'test_D_age.txt')
			total_list_test += file_reader(root_dir_id + 'test_E_ill.txt')
			self.spoof_train_fname = []
			self.spoof_test_fname  = []
			self.live_train_fname  = []
			self.live_test_fname   = []
			for _ in total_list_train:
				if 'Live' in _:
					self.live_train_fname.append(_)
				else:
					self.spoof_train_fname.append(_)
			for _ in total_list_test:
				if 'Live' in _:
					self.live_test_fname.append(_)
				else:
					self.spoof_test_fname.append(_)

	# overriding the compile method.
	def compile(self, dataset_name='SiWM-v2'):
		'''generates train and test list for SIW-Mv2.'''
		# Train data.
		for x in self.live_train_fname:
			if x != '':
				self.LI_DATA_DIR.append(self.live_img_root + x)
		for x in self.spoof_train_fname:
			if x != '':
				if self.protocol in [1, 3]:
					self.SP_DATA_DIR.append(self.spoof_img_root + x)
				elif self.protocol == 2:
					if self.spoof_type_dict[self.unknown] not in x:
						self.SP_DATA_DIR.append(self.spoof_img_root + x)

		for x in self.live_test_fname:
			if x != '':
				self.LI_DATA_DIR_TEST.append(self.live_img_root + x)
		if self.protocol in [1, 3]:
			for x in self.spoof_test_fname:
				if x != '':
					self.SP_DATA_DIR_TEST.append(self.spoof_img_root + x)
		elif self.protocol == 2:
			self.SP_DATA_DIR_TEST = glob(self.spoof_img_root + self.spoof_type_dict[self.unknown] + '_*')

class Config_custom(Config):
	def __init__(self, args):
		super().__init__(args)
		self.dataset = 'custom'
		self.BATCH_SIZE = 4
		self.root_dir = './preprocessed_image_train'
		self.phase = 'ft'
		# self.partition_folder = args.partition
		self.SP_DATA = glob(self.root_dir + '/spoof/*')
		self.LI_DATA = glob(self.root_dir + '/live/*')

		# def read_txt_file(file_name):
		# 	sub_list = []
		# 	txt_file = os.path.join(self.partition_folder, file_name)
		# 	f = open(txt_file, 'r')
		# 	lines = f.readlines()
		# 	category = 'live' if 'live' in file_name else 'spoof'
		# 	for _ in lines:
		# 		if _ != "":
		# 			_ = _.strip()
		# 			sub_list.append(os.path.join(self.root_dir, category, _))
		# 	return sub_list

		# self.LI_DATA_DIR = read_txt_file('train_target_live.txt')
		# self.SP_DATA_DIR = read_txt_file('train_target_spoof.txt')
		# self.LI_DATA_DIR_TEST = read_txt_file('test_source_live.txt')
		# self.SP_DATA_DIR_TEST = read_txt_file('test_source_spoof.txt')
		# self.LI_DATA_DIR_TEST_B = read_txt_file('test_target_live.txt')
		# self.SP_DATA_DIR_TEST_B = read_txt_file('test_target_spoof.txt')

  #       # This is a misuse of assertions. These should be tests and exceptions as
  #       # assertions can be disabled at runtime.  Sigh.
		# assert len(self.LI_DATA_DIR) != 0
		# assert len(self.SP_DATA_DIR) != 0
		# assert len(self.LI_DATA_DIR_TEST) != 0
		# assert len(self.SP_DATA_DIR_TEST) != 0
		# assert len(self.LI_DATA_DIR_TEST_B) != 0
		# assert len(self.SP_DATA_DIR_TEST_B) != 0

		self.inference_data_dir = args.dir
		self.inference_data_img = args.img

		if self.inference_data_dir == None and self.inference_data_img == None:
			assert False, print("Please offer either the inference image or inference image folder.")
		else:
			if self.inference_data_dir != None:
				if not os.path.isdir(self.inference_data_dir):
					assert False, print("Please offer a valid directory.")
			elif self.inference_data_img != None:
				image_suffix = self.inference_data_img.lower()
				if not image_suffix.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
					assert False, print("Please offer a valid image file.")
