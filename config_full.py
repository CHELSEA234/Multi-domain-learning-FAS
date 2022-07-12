import tensorflow as tf
import os
import glob
import abc
import csv

# Configuration class.
class Config(object):
	LOG_DEVICE_PLACEMENT = False
	IMG_SIZE = 256
	MAP_SIZE = 32
	FIG_SIZE = 128	

	# Training meta.
	STEPS_PER_EPOCH = 2000
	IMG_LOG_FR = 100
	TXT_LOG_FR = 1000
	# Epochs after which learning rate decays.
	NUM_EPOCHS_PER_DECAY = 10.0   

	# Initial learning rate.
	LEARNING_RATE = 1e-4   
	# Learning rate decay factor.       
	LEARNING_RATE_DECAY_FACTOR = 0.89  
	# The decay to use for the moving average.
	LEARNING_MOMENTUM = 0.999  
	# The decay to use for the moving average.   
	MOVING_AVERAGE_DECAY = 0.9999     
	GAN = 'ls' # 'hinge', 'ls'
	DECAY_STEP = 1

	# Discriminator depth.
	n_layer_D = 4

	def __init__(self, args):
		self.MAX_EPOCH = args.epoch
		self.GPU_INDEX = args.cuda
		self.phase = args.stage
		assert self.phase in ['pretrain', 'ft', 'ub'], print("Please offer the valid phase!")
		self.type  = args.type
		self.SET = args.set
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
		# pass
		return

	@abc.abstractmethod
	def search_folder_wrapper(self, root_dir, filenames):
		# pass
		return

	def compile(self, dataset_name='SiWM-v2'):
		protocol_type = self.SET
		assert dataset_name in ['SiWM-v2', 'SiW', 'Oulu'], print("Please offer the correct dataset.")

		# Training data.
		counter = 0
		if self.phase == 'pretrain':
			with open(self.pretrain_train, 'r') as f:
				filenames = f.read().split('\n')
			# filenames = filenames[:int(len(filenames)/3)+1]
		elif self.phase == 'ft':
			with open(self.type_train, 'r') as f:
				filenames = f.read().split('\n')
			# filenames = filenames[:int(len(filenames)/3)+1]
		elif self.phase == 'ub':
			with open(self.pretrain_train, 'r') as f:
				filenames0 = f.read().split('\n')	
			with open(self.type_train, 'r') as f:
				filenames1 = f.read().split('\n')

			# ## oulu A is 21, B, C and D are around 6.
			# ## siw A is 80, B, C and D are 15, 13 and 7.
			# filenames0 = filenames0[:int(len(filenames0)/3)+1]
			# filenames1 = filenames1[:int(len(filenames1)/3)+1]

			# filenames0 = filenames0 * 10
			# filenames1 = filenames1 * 50
			# filenames0 = filenames0[:200]
			# filenames1 = filenames1[:200]
			# assert len(filenames0) == len(filenames0), print("In the upperbound, A and B should be balanced.")
			filenames = filenames0 + filenames1

		def filter_ill_siw(old_list=[], state='pretrain'):
			new_list = []
			## GX: each separation has different portions of SIW and SIWM-v2
			if state == 'pretrain':
				target_list = ['0', '1', '2', '3']
			elif state == 'ft':
				target_list = ['4']
			else:
				assert False
			for _ in old_list:
				sub_id = _.split('/')[-1]
				label_cur = illu_dict[sub_id]
				if label_cur in target_list:
					new_list.append(_)
			return new_list

		def filter_ill_oulu(old_list=[], state='pretrain'):
			new_list = []
			if state == 'pretrain':
				target_list = ['1', '2']
			elif state == 'ft':
				target_list = ['3']
			else:
				# target_list = ['1','2','3']	# temporary useage.
				assert False
			for _ in old_list:
				device_id, sess_id, sub_id, sp_id = _.split('/')[-1].split('_')
				if sess_id in target_list:
					new_list.append(_)
				else:
					if sub_id in ['5','10','15','20','25','30','35','40','50','60']:	# these subject contains session#3.
						new_list.append(_)
			return new_list
			
		# print(f"=====================================")
		# print(f"Compiling the {dataset_name} dataset.")
		# print(f"=====================================")
		if dataset_name in ["SiW","SiWM-v2"]:
			csv_file_name = "/user/guoxia11/cvl/anti_spoofing/illumination_estimation/DPR/combine_label_illu.csv"
			csv_file = open(csv_file_name)
			csv_reader = csv.reader(csv_file, delimiter=',')
			line_count  = 0
			illu_dict = dict()
			illu_dict_siwm = dict()
			for row in csv_reader:
				if line_count == 0:
					pass
				else:
					if "siwm" not in row[0]:
						sub_id, label = row[0].split('/')[-2], row[1]
						illu_dict[sub_id] = label
					else:
						sub_id, label = row[0].split('/')[-2], row[1]
						illu_dict_siwm[sub_id] = label
				line_count += 1
			csv_file.close()

		if self.phase=='pretrain':
			## the new version pretrain has all 1,2 but small amount of 3.
			# print("==========================")
			self.LI_DATA_DIR, self.SP_DATA_DIR = self.search_folder_wrapper(self.root_dir, filenames)
			# print(len(self.LI_DATA_DIR), len(self.SP_DATA_DIR))
			if dataset_name != 'SiWM-v2':
				if dataset_name == 'Oulu':
					self.LI_DATA_DIR = filter_ill_oulu(self.LI_DATA_DIR, state='pretrain')
					self.SP_DATA_DIR = filter_ill_oulu(self.SP_DATA_DIR, state='pretrain')
				elif dataset_name == 'SiW':
					self.LI_DATA_DIR = filter_ill_siw(self.LI_DATA_DIR, state='pretrain')
					self.SP_DATA_DIR = filter_ill_siw(self.SP_DATA_DIR, state='pretrain')
			# import sys;sys.exit(0)
			# print(len(self.LI_DATA_DIR), len(self.SP_DATA_DIR))
			# print("==========================")
			# import sys;sys.exit(0)
		elif self.phase=='ft':
			## the new version ft in ill, it combines BCD, with small amount of 1,2, most 3.
			# print("==========================")
			self.LI_DATA_DIR, self.SP_DATA_DIR = self.search_folder_wrapper(self.root_dir, filenames)
			# print(len(self.LI_DATA_DIR), len(self.SP_DATA_DIR))
			if dataset_name != 'SiWM-v2' and self.type=='illu':
				if dataset_name == 'Oulu':
					self.LI_DATA_DIR = filter_ill_oulu(self.LI_DATA_DIR, state='ft')
					self.SP_DATA_DIR = filter_ill_oulu(self.SP_DATA_DIR, state='ft')
				elif dataset_name == 'SiW':
					self.LI_DATA_DIR = filter_ill_siw(self.LI_DATA_DIR, state='ft')
					self.SP_DATA_DIR = filter_ill_siw(self.SP_DATA_DIR, state='ft')
		elif self.phase=='ub':
			new_li = []
			new_sp = []
			self.LI_DATA_DIR, self.SP_DATA_DIR = self.search_folder_wrapper(self.root_dir, filenames0)
			if dataset_name != 'SiWM-v2':
				if dataset_name == 'Oulu':
					new_li = filter_ill_oulu(self.LI_DATA_DIR, state='pretrain')
					new_sp = filter_ill_oulu(self.SP_DATA_DIR, state='pretrain')
				elif dataset_name == 'SiW':
					new_li = filter_ill_siw(self.LI_DATA_DIR, state='pretrain')
					new_sp = filter_ill_siw(self.SP_DATA_DIR, state='pretrain')
			self.LI_DATA_DIR, self.SP_DATA_DIR = self.search_folder_wrapper(self.root_dir, filenames1)
			if dataset_name != 'SiWM-v2' and self.type=='illu':
				if dataset_name == 'Oulu':
					new_li += filter_ill_oulu(self.LI_DATA_DIR, state='ft')
					new_sp += filter_ill_oulu(self.SP_DATA_DIR, state='ft')
				elif dataset_name == 'SiW':
					new_li += filter_ill_siw(self.LI_DATA_DIR, state='ft')
					new_sp += filter_ill_siw(self.SP_DATA_DIR, state='ft')
			else:
				new_li += self.LI_DATA_DIR
				new_sp += self.SP_DATA_DIR
			self.LI_DATA_DIR = new_li
			self.SP_DATA_DIR = new_sp
			# print(len(self.LI_DATA_DIR), len(self.SP_DATA_DIR))
		else:
			assert False, print("Please offer a right phase to work.")

		## the new version code does not need to do any chnage in test set.
		# Val/Test data.
		with open(self.pretrain_test, 'r') as f:
			filenames = f.read().split('\n')
		self.LI_DATA_DIR_TEST, self.SP_DATA_DIR_TEST = self.search_folder_wrapper(self.root_dir, 
																				  filenames)
		with open(self.type_test, 'r') as f:
			filenames = f.read().split('\n')
		self.LI_DATA_DIR_TEST_B, self.SP_DATA_DIR_TEST_B = self.search_folder_wrapper(self.root_dir,
																					  filenames)

	def filename_gen(self):
		# TODO: one function finds the filenames.
		return 

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

	def search_folder(self, root_dir, sub_id, stype):
		super(Config_oulu, self).search_folder(root_dir, sub_id, stype)
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

	def search_folder_wrapper(self, root_dir, filenames):
		super(Config_oulu, self).search_folder_wrapper(root_dir, filenames)
		li_list, sp_list = [], []
		for x in filenames:
			if x in ["0", ""]:
				continue
			else:
				digit_len = len(x)
				if digit_len == 1:
					sub_id = '0'+x
				else:
					sub_id = x
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

	def search_folder(self, root_dir, sub_id, stype):
		super(Config_siw, self).search_folder(root_dir, sub_id, stype)
		if stype == 'Live':
			folder_list = glob.glob(root_dir+f'train/live/{sub_id}*')
			folder_list += glob.glob(root_dir+f'test/live/{sub_id}*')
		elif stype == 'Spoof':
			folder_list = glob.glob(root_dir+f'train/spoof/{sub_id}*')		
			folder_list += glob.glob(root_dir+f'test/spoof/{sub_id}*')
		else:
			assert False, print("Please offer a valid stype here.")
		return folder_list

	def search_folder_wrapper(self, root_dir, filenames):
		super(Config_siw, self).search_folder_wrapper(root_dir, filenames)
		li_list, sp_list = [], []
		for x in filenames:
			if x in ["0", ""]:
				continue
			else:
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
		if self.type == 'age':
			self.type_train = root_dir_id + "age_list/list/age_B_train_ub.txt"
			self.type_test  = root_dir_id + "age_list/list/age_B_test.txt"
		elif self.type == 'spoof':
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
		self.pretrain_train = root_dir_id + 'spoof_type_list/pretrain_A_train_balanced.txt'
		self.pretrain_test  = root_dir_id + 'spoof_type_list/pretrain_A_test.txt'

		csv_file_name = "/user/guoxia11/cvl/anti_spoofing/illumination_estimation/DPR/combine_label_illu.csv"
		csv_file = open(csv_file_name)
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count  = 0
		self.illu_dict = dict()
		self.illu_dict_siwm = dict()
		for row in csv_reader:
			if line_count == 0:
				pass
			else:
				if "siwm" not in row[0]:
					sub_id, label = row[0].split('/')[-2], row[1]
					self.illu_dict[sub_id] = label
				else:
					sub_id, label = row[0].split('/')[-2], row[1]
					self.illu_dict_siwm[sub_id] = label
			line_count += 1
		csv_file.close()

	def filter_ill_siwm(self, old_list=[], state='pretrain'):
		new_list = []
		## GX: each separation has different portions of SIW and SIWM-v2
		if state == 'pretrain':
			target_list = ['0', '1', '2', '3']
		elif state == 'ft':
			target_list = ['4']
		else:
			assert False
		for _ in old_list:
			if _ != "":
				label_cur = self.illu_dict_siwm["siwm_"+_]
				if label_cur in target_list:
					new_list.append(_)
		return new_list

	def compile_siwm(self):
		# super(Config_siwm, self).compile()
		# Training data.
		self.SP_DATA_DIR, self.LI_DATA_DIR = [], []
		if self.phase == 'pretrain':
			with open(self.pretrain_train, 'r') as f:
				filenames = f.read().split('\n')
			# print(len(filenames))
			filenames = self.filter_ill_siwm(filenames,state='pretrain')
			# print(len(filenames))
		elif self.phase == 'ft':
			with open(self.type_train, 'r') as f:
				filenames = f.read().split('\n')
			# print(len(filenames))
			if self.type == 'illu':
				filenames = self.filter_ill_siwm(filenames,state='ft')
			# print(len(filenames))
			# import sys;sys.exit(0)
		elif self.phase == 'ub':
			with open(self.pretrain_train, 'r') as f:
				filenames = f.read().split('\n')
			# print(len(filenames))
			new_pretrain = self.filter_ill_siwm(filenames,state='pretrain')
			# print(len(new_pretrain))
			with open(self.type_train, 'r') as f:
				filenames = f.read().split('\n')
			# print("===========================")
			# print(len(filenames))
			if self.type == 'illu':
				filenames = self.filter_ill_siwm(filenames,state='ft')
			# print(len(filenames))
			filenames = new_pretrain + filenames

		for x in filenames:
			if x == '':
				continue
			elif 'Live' not in x:
				self.SP_DATA_DIR.append('/user/guoxia11/cvlshare/cvl-guoxia11/Spoof/'+x)
			else:
				self.LI_DATA_DIR.append('/user/guoxia11/cvlshare/cvl-guoxia11/Live/'+x)

		# Val/Test data.
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

		# Val/Test data.
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
