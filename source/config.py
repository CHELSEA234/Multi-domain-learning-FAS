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
import glob
import abc
import csv
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
	NUM_EPOCHS_PER_DECAY = 10.0  		# Epochs after which learning rate decays.
	LEARNING_RATE_DECAY_FACTOR = 0.89  	# The decay to use for the moving average.
	LEARNING_MOMENTUM = 0.999   
	MOVING_AVERAGE_DECAY = 0.9999		# The decay to use for the moving average.       
	GAN = 'ls' # 'hinge', 'ls'
	DECAY_STEP = 1
	n_layer_D = 4

	def __init__(self, args):
		self.MAX_EPOCH = args.epoch
		self.GPU_INDEX = args.cuda
		self.phase = args.stage
		assert self.phase in ['pretrain', 'ft', 'ub'], print("Please offer the valid phase!")
		self.type  = args.type
		self.SET = args.set
		self.illu_dict = dict()
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

	def _filter_ill_siw(self, old_list=[], state='pretrain'):
		'''assigns siw to different illumination.'''
		new_list = []
		assert state in ['pretrain', 'ft'], print("Please offer the right stage.")
		target_list = ['0', '1', '3', '4'] if state == 'pretrain' else ['2']
		for _ in old_list:
			sub_id, label_cur = _.split('/')[-1], self.illu_dict[sub_id]
			if label_cur in target_list:
				new_list.append(_)
		return new_list

	def _filter_ill_oulu(self, old_list=[], state='pretrain'):
		'''assigns oulu to different illumination.'''
		new_list = []
		assert state in ['pretrain', 'ft'], print("Please offer the right stage.")
		target_list = ['1', '2'] if state == 'pretrain' else ['3']
		sess_3_subs = ['5','10','15','20','25','30','35','40','50','60'] # session#3 subject.
		for _ in old_list:
			device_id, sess_id, sub_id, sp_id = _.split('/')[-1].split('_')
			if (sess_id in target_list) or (sub_id not in target_list and sub_id in sess_3_subs):
				new_list.append(_)
		return new_list

	def illu_list_gen(self, li_list, sp_list, dataset_name, state):
		'''calls fuctions to assign either oulu or siw into different illuminations.'''
		if dataset_name == 'Oulu':
			return self._filter_ill_oulu(li_list, state), self._filter_ill_oulu(sp_list, state)
		elif dataset_name == 'SiW':
			return self._filter_ill_siw(li_list, state), self._filter_ill_siw(sp_list, state)
		else:
			return 

	def _construct_ill_dict(self, dataset_name):
		'''each subject is associated with one illumination.'''
		csv_file_name = "/user/guoxia11/cvl/anti_spoofing/illumination_estimation/DPR/combine_label_illu.csv"
		csv_file = open(csv_file_name)
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count  = 0
		for row in csv_reader:
			if line_count != 0 and "siwm" not in row[0]:
					sub_id, label = row[0].split('/')[-2], row[1]
					self.illu_dict[sub_id] = label
			line_count += 1
		csv_file.close()

	def compile(self, dataset_name='SiW'):
		'''generates train and test list for SIW and Oulu.'''
		assert dataset_name in ['SiW', 'Oulu'], print("Please offer the correct dataset.")
		# Training data.
		#########################################################
		## In Oulu subject # in A is 21, B, C and D are around 6.
		## In SiW subject # in A, B, C and D are 80, 15, 13 and 7.
		if self.phase == 'pretrain':
			filenames = file_reader(self.pretrain_train)
		elif self.phase == 'ft':
			filenames = file_reader(self.type_train)
		elif self.phase == 'ub':
			filenames0 = file_reader(self.pretrain_train)
			filenames1 = file_reader(self.type_train)
			filenames = filenames0 + filenames1
		if dataset_name == 'SiWM-v2':
 			self._construct_ill_dict(dataset_name)

		if self.phase=='pretrain':
			## the new version pretrain has all 1,2 but small amount of 3.
			self.LI_DATA_DIR, self.SP_DATA_DIR = self.search_folder_wrapper(self.root_dir, filenames)
			self.LI_DATA_DIR, self.SP_DATA_DIR = self.illu_list_gen(self.LI_DATA_DIR, self.SP_DATA_DIR, dataset_name, state='pretrain')
		elif self.phase=='ft':
			## the new version ft in ill, it combines BCD, with small amount of 1,2, most 3.
			self.LI_DATA_DIR, self.SP_DATA_DIR = self.search_folder_wrapper(self.root_dir, filenames)
			if self.type == 'illu':
				self.LI_DATA_DIR, self.SP_DATA_DIR = self.illu_list_gen(self.LI_DATA_DIR, self.SP_DATA_DIR, dataset_name, state='ft')
		elif self.phase=='ub':
			new_li, new_sp = [], []
			self.LI_DATA_DIR, self.SP_DATA_DIR = self.search_folder_wrapper(self.root_dir, filenames0)
			new_li_pre, new_sp_pre = self.illu_list_gen(self.LI_DATA_DIR, self.SP_DATA_DIR, dataset_name, state='pretrain')
			self.LI_DATA_DIR, self.SP_DATA_DIR = self.search_folder_wrapper(self.root_dir, filenames1)
			if self.type=='illu':
				new_li_ft, new_sp_ft = self.illu_list_gen(self.LI_DATA_DIR, self.SP_DATA_DIR, dataset_name, state='ft')
			else:
				new_li_ft, new_sp_ft = self.LI_DATA_DIR, self, SP_DATA_DIR
			self.LI_DATA_DIR = new_li_pre + new_li_ft
			self.SP_DATA_DIR = new_sp_pre + new_sp_ft
		else:
			assert False, print("Please offer a right phase to work.")

		# Val/Test data.
		with open(self.pretrain_test, 'r') as f:
			filenames = f.read().split('\n')
		self.LI_DATA_DIR_TEST, self.SP_DATA_DIR_TEST = self.search_folder_wrapper(self.root_dir, filenames)
		with open(self.type_test, 'r') as f:
			filenames = f.read().split('\n')
		self.LI_DATA_DIR_TEST_B, self.SP_DATA_DIR_TEST_B = self.search_folder_wrapper(self.root_dir, filenames)

class Config_oulu(Config):
	def __init__(self, args):
		super().__init__(args)
		self.dataset = 'oulu'
		self.BATCH_SIZE = 1
		self.root_dir = "/user/guoxia11/cvlshare/Databases/Oulu/bin/"
		root_dir_id = '/user/guoxia11/cvl/anti_spoofing/stats_update/oulu_datalist/'	
		self.pretrain_train = root_dir_id + 'A_train_oulu.txt'
		self.pretrain_test  = root_dir_id + 'A_test_oulu.txt' 
		if self.type == 'age':
			self.type_train = root_dir_id + 'C_train_oulu.txt'
			self.type_test  = root_dir_id + 'C_test_oulu.txt'
		elif self.type == 'spoof':
			self.type_train = root_dir_id + 'B_train_oulu.txt'
			self.type_test  = root_dir_id + 'B_test_oulu.txt'
		elif self.type == 'race':
			self.type_train = root_dir_id + 'D_train_oulu.txt'
			self.type_test  = root_dir_id + 'D_test_oulu.txt'
		elif self.type == 'illu':
			self.type_train = root_dir_id + 'E_train_oulu.txt'
			self.type_test  = root_dir_id + 'E_test_oulu.txt'
		else:
			assert False, print("wait to implement...")

	# overriding abstract method
	def search_folder(self, root_dir, sub_id, stype):
		if stype == 'Live':
			folder_list = glob.glob(root_dir+f'train/live/*{sub_id}*')
			folder_list += glob.glob(root_dir+f'eval/live/*{sub_id}*')
			folder_list += glob.glob(root_dir+f'test/live/*{sub_id}*')
		elif stype == 'Spoof':
			folder_list = glob.glob(root_dir+f'train/spoof/*{sub_id}*')		
			folder_list += glob.glob(root_dir+f'eval/spoof/*{sub_id}*')
			folder_list += glob.glob(root_dir+f'test/spoof/*{sub_id}*')
		else:
			assert False, print("Please offer a valid stype here.")
		return folder_list

	# overriding abstract method
	def search_folder_wrapper(self, root_dir, filenames):
		li_list, sp_list = [], []
		for x in filenames:
			if x not in ["0", ""]:
				sub_id = '0'+x if len(x) == 1 else x
				li_list += self.search_folder(root_dir=root_dir, sub_id=sub_id, stype="Live")
				sp_list += self.search_folder(root_dir=root_dir, sub_id=sub_id, stype="Spoof")
		return li_list, sp_list

class Config_siw(Config):
	def __init__(self, args):
		super().__init__(args)
		self.dataset = "SiW"
		self.BATCH_SIZE = 1
		root_dir_id = '/user/guoxia11/cvl/anti_spoofing/stats_update/SiW_datalist/'
		self.root_dir = "/user/guoxia11/cvlshare/Databases/SiW/bin/"
		self.pretrain_train = root_dir_id + 'A_train_sub_id_siw.txt'
		self.pretrain_test  = root_dir_id + 'A_test_sub_id_siw.txt' 
		if self.type == 'age':
			self.type_train = root_dir_id + 'C_train_sub_id_siw.txt'
			self.type_test  = root_dir_id + 'C_test_sub_id_siw.txt'
		elif self.type == 'spoof':
			self.type_train = root_dir_id + 'B_train_sub_id_siw.txt'
			self.type_test  = root_dir_id + 'B_test_sub_id_siw.txt'
		elif self.type == 'race':
			self.type_train = root_dir_id + 'D_train_sub_id_siw.txt'
			self.type_test  = root_dir_id + 'D_test_sub_id_siw.txt'
		elif self.type == 'illu':
			self.type_train = root_dir_id + 'E_train_sub_id_siw.txt'
			self.type_test  = root_dir_id + 'E_test_sub_id_siw.txt'
		else:
			assert False, print("wait to implement...")

	# overriding abstract method
	def search_folder(self, root_dir, sub_id, stype):
		if stype == 'Live':
			folder_list = glob.glob(root_dir+f'train/live/{sub_id}*')
			folder_list += glob.glob(root_dir+f'test/live/{sub_id}*')
		elif stype == 'Spoof':
			folder_list = glob.glob(root_dir+f'train/spoof/{sub_id}*')		
			folder_list += glob.glob(root_dir+f'test/spoof/{sub_id}*')
		else:
			assert False, print("Please offer a valid stype here.")
		return folder_list

	# overriding abstract method
	def search_folder_wrapper(self, root_dir, filenames):
		li_list, sp_list = [], []
		for x in filenames:
			if x not in ["0", ""]:
				digit_len = len(x)
				if digit_len == 1:
					sub_id = '00'+x
				elif digit_len == 2:
					sub_id = '0'+x
				else:
					sub_id = x
				li_list += self.search_folder(root_dir=root_dir, sub_id=sub_id, stype="Live")
				sp_list += self.search_folder(root_dir=root_dir, sub_id=sub_id, stype="Spoof")
		return li_list, sp_list

class Config_siwm(Config):
	def __init__(self, args):
		super().__init__(args)
		self.dataset = "SiWM-v2"
		self.BATCH_SIZE = 2
		root_dir_id = "/user/guoxia11/cvl/anti_spoofing/"
		## 1707 samples, with 940 subjects; balanced among subjects.
		self.pretrain_train = root_dir_id + 'spoof_type_list/pretrain_A_train_balanced.txt'
		self.pretrain_test  = root_dir_id + 'spoof_type_list/pretrain_A_test.txt'
		if self.type == 'age':
			self.type_train = root_dir_id + "age_list/list/age_B_train_ub.txt"
			self.type_test  = root_dir_id + "age_list/list/age_B_test.txt"
		elif self.type == 'spoof':
			## 1707 samples, with 65 subjects; balanced among subjects.
			self.type_train = root_dir_id + "spoof_type_list/B_train_spoof_balanced_ub.txt"
			self.type_test  = root_dir_id + "spoof_type_list/B_test_spoof.txt"
		elif self.type == 'race':
			self.type_train = root_dir_id + "race_list/race_small_B_train_ub.txt"
			self.type_test  = root_dir_id + "race_list/race_B_test.txt"
		elif self.type == 'illu':
			self.type_train = root_dir_id + "age_list/list/ill_E_train_ub.txt"
			self.type_test  = root_dir_id + "age_list/list/ill_E_test.txt"
		else:
			assert False, print("wait to implement...")

	# overriding the compile method.
	def compile(self, dataset_name='SiWM-v2'):
		'''generates train and test list for SIW-Mv2.'''
		# Train data.
		## GX: compile_siwm does not have filter_out process for the new illumination.
		self.SP_DATA_DIR, self.LI_DATA_DIR = [], []
		if self.phase == 'pretrain':
			with open(self.pretrain_train, 'r') as f:
				filenames = f.read().split('\n')
		elif self.phase == 'ft':
			with open(self.type_train, 'r') as f:
				filenames = f.read().split('\n')
		elif self.phase == 'ub':
			with open(self.pretrain_train, 'r') as f:
				filenames = f.read().split('\n')
			new_pretrain = filenames
			with open(self.type_train, 'r') as f:
				filenames = f.read().split('\n')
			filenames = new_pretrain + filenames
		for x in filenames:
			if x == '':
				continue
			elif 'Live' not in x:
				self.SP_DATA_DIR.append('/user/guoxia11/cvlshare/cvl-guoxia11/Spoof/'+x)
			else:
				self.LI_DATA_DIR.append('/user/guoxia11/cvlshare/cvl-guoxia11/Live/'+x)
		# Test_A data.
		self.SP_DATA_DIR_TEST, self.LI_DATA_DIR_TEST = [], []
		with open(self.pretrain_test, 'r') as f:
			filenames = f.read().split('\n')
		for x in filenames:
			if x == '':
				continue
			elif 'Live' not in x:
				self.SP_DATA_DIR_TEST.append('/user/guoxia11/cvlshare/cvl-guoxia11/Spoof/'+x)
			else:
				self.LI_DATA_DIR_TEST.append('/user/guoxia11/cvlshare/cvl-guoxia11/Live/'+x)
		# Test_B data.
		self.SP_DATA_DIR_TEST_B, self.LI_DATA_DIR_TEST_B = [], []
		with open(self.type_test, 'r') as f:
			filenames = f.read().split('\n')
		for x in filenames:
			if x == '':
				continue
			elif 'Live' not in x:
				self.SP_DATA_DIR_TEST_B.append('/user/guoxia11/cvlshare/cvl-guoxia11/Spoof/'+x)
			else:
				self.LI_DATA_DIR_TEST_B.append('/user/guoxia11/cvlshare/cvl-guoxia11/Live/'+x)
