from metrics_ import my_metrics
import tensorflow as tf
import csv
import argparse
import os
import time
import math
import numpy as np
from tqdm import tqdm
from model import Generator, Discriminator, region_estimator, Multi_Con_Discriminator, Feature_transform_layer
from dataset_full import Dataset
from utils import l1_loss, l2_loss, hinge_loss, Logging
from config_full import Config_siwm as Config
from config_full import Config_siw, Config_oulu
from tensorboardX import SummaryWriter

class STDNet(object):
	def __init__(self, config, config_siw, config_oulu, args):
		self.config = config
		self.config_siw  = config_siw
		self.config_oulu = config_oulu
		self.lr = config.lr
		self.bs = config.BATCH_SIZE + config_siw.BATCH_SIZE + config_oulu.BATCH_SIZE
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
		self.disc_kd = Multi_Con_Discriminator()
		self.layer_1 = Feature_transform_layer(64)
		self.layer_2 = Feature_transform_layer(40)
		self.gen_opt = tf.keras.optimizers.Adam(self.lr)
		self.disc_opt = tf.keras.optimizers.Adam(self.lr)

		# Checkpoint initialization.
		self.save_dir = config.save_model_dir
		self.checkpoint_path_g = self.save_dir+"/gen/cp-{epoch:04d}.ckpt"
		self.checkpoint_path_re= self.save_dir+"/ReE/cp-{epoch:04d}.ckpt"
		self.checkpoint_path_kd= self.save_dir+"/kdd/cp-{epoch:04d}.ckpt"
		self.checkpoint_path_l1= self.save_dir+"/l1/cp-{epoch:04d}.ckpt"
		self.checkpoint_path_l2= self.save_dir+"/l2/cp-{epoch:04d}.ckpt"
		self.checkpoint_path_d1= self.save_dir+"/dis1/cp-{epoch:04d}.ckpt"
		self.checkpoint_path_d2= self.save_dir+"/dis2/cp-{epoch:04d}.ckpt"
		self.checkpoint_path_d3= self.save_dir+"/dis3/cp-{epoch:04d}.ckpt"
		self.checkpoint_path_g_op = self.save_dir+"/g_opt/cp-{epoch:04d}.ckpt"
		self.checkpoint_path_d_op = self.save_dir+"/d_opt/cp-{epoch:04d}.ckpt"
		
		self.checkpoint_dir_g    = os.path.dirname(self.checkpoint_path_g)
		self.checkpoint_dir_re   = os.path.dirname(self.checkpoint_path_re)
		self.checkpoint_dir_kd   = os.path.dirname(self.checkpoint_path_kd)
		self.checkpoint_dir_l1   = os.path.dirname(self.checkpoint_path_l1)
		self.checkpoint_dir_l2   = os.path.dirname(self.checkpoint_path_l2)
		self.checkpoint_dir_d1   = os.path.dirname(self.checkpoint_path_d1)
		self.checkpoint_dir_d2   = os.path.dirname(self.checkpoint_path_d2)
		self.checkpoint_dir_d3   = os.path.dirname(self.checkpoint_path_d3)
		self.checkpoint_dir_g_op = os.path.dirname(self.checkpoint_path_g_op)
		self.checkpoint_dir_d_op = os.path.dirname(self.checkpoint_path_d_op)

		self.model_list  = [self.gen, self.RE, self.disc_kd, 
							self.layer_1, self.layer_2,
							self.disc1, self.disc2, self.disc3]
		self.model_p_list= [self.checkpoint_path_g, 
							self.checkpoint_path_re,
							self.checkpoint_path_kd,
							self.checkpoint_path_l1,
							self.checkpoint_path_l2,
							self.checkpoint_path_d1,
							self.checkpoint_path_d2,
							self.checkpoint_path_d3]
		self.model_d_list= [self.checkpoint_dir_g,
							self.checkpoint_dir_re,
							self.checkpoint_dir_kd,
							self.checkpoint_dir_l1,
							self.checkpoint_dir_l2,
							self.checkpoint_dir_d1,
							self.checkpoint_dir_d2,
							self.checkpoint_dir_d3]

		# Log class for displaying the losses.
		self.log = Logging(config)
		self.gen_opt  = tf.keras.optimizers.Adam(self.lr)
		self.disc_opt = tf.keras.optimizers.Adam(self.lr)

	def update_lr(self, new_lr=0, restore=False, last_epoch=0):
		if restore:
			assert last_epoch != 0, print("Restoring LR should not start at 0 epoch.")
			self.lr = self.lr * np.power(self.config.LEARNING_RATE_DECAY_FACTOR, last_epoch)
			print(f"Restoring the previous learning rate {self.lr} at epoch {last_epoch}.")
			## TODO: the recovery should also be done on the m_t and v_t.
			## TODO: https://ruder.io/optimizing-gradient-descent/
		self.gen_opt.learning_rate.assign(self.lr)	
		self.disc_opt.learning_rate.assign(self.lr)

	def _restore(self, model, checkpoint_dir, pretrain=False):
		last_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
		model.load_weights(last_checkpoint)
		if not pretrain:
			last_epoch = int((last_checkpoint.split('.')[1]).split('-')[-1])
			return last_epoch

	def _save(self, model, checkpoint_path, epoch):
		model.save_weights(checkpoint_path.format(epoch=epoch))

	#############################################################################
	def train(self, dataset, dataset_siw, dataset_oulu, config):
		self.csv_file = open(self.config.csv_file_name, mode='w')
		self.csv_writer = csv.writer(self.csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		self.csv_writer.writerow(['Video name', 'Dataset', 
								  'p_pretrain', 'dmap_pred_pretrain', 'region_map_pretrain', 
								  'p', 'dmap_pred', 'region_map',
								  'Label', 'test_mode'])
		epoch_num = self.config.epoch_num_pick
		self.log.epoch = epoch_num
		for model_, model_dir_ in zip(self.model_list, self.model_d_list):
			epoch_suffix = f"/cp-{epoch_num:04d}.ckpt"
			current_checkpoint = model_dir_ + epoch_suffix
			model_.load_weights(current_checkpoint)
		print(f"Finish loading {epoch_num}th checkpoint.")
		start = time.time()
		for test_mode in ["test_A", "test_B"]:
			self.test_step(test_mode)
			_, final_score2list, _, _, label_list = self.test_step(test_mode)
			print ('\n*****Time for epoch {} is {} sec*****'.format(epoch_num + 1, 
																	int(time.time()-start)))

	def test_step(self, test_mode, viz_mode=False):
		## 147 1102 900
		# dataset_test = Dataset(self.config, test_mode)
		# # print(len(dataset_test), dataset_test._info())
		# dataset_test = Dataset(self.config_siw, test_mode)
		# # print(len(dataset_test), dataset_test._info())
		# dataset_test = Dataset(self.config_oulu, test_mode)
		# # print(len(dataset_test), dataset_test._info())
		# # import sys;sys.exit(0)
		score00, score20, score30, score40, label_lst0 = self.test_step_helper(self.config, 
																				test_mode,
																				viz_mode,
																				prefix_dataset='SIWM')  
		score01, score21, score31, score41, label_lst1 = self.test_step_helper(self.config_siw, 
																				test_mode,
																				viz_mode,
																				prefix_dataset='SIW')
		score02, score22, score32, score42, label_lst2 = self.test_step_helper(self.config_oulu,
																				test_mode,
																				viz_mode,
																				prefix_dataset='Oulu')
		final_score0list = score00 + score01 + score02
		final_score2list = score20 + score21 + score22
		final_score3list = score30 + score31 + score32
		final_score4list = score40 + score41 + score42
		label_list = label_lst0 + label_lst1 + label_lst2
		# print("the total length is: ", len(label_list))
		return final_score0list, final_score2list, final_score3list, final_score4list, label_list 

	def test_step_helper(self, config_cur, test_mode, prefix_dataset=None):
		old_bs = config_cur.BATCH_SIZE
		# config_cur.BATCH_SIZE = 8 # GX: in the training, bs=4, then sp_bs+li_bs = 8.
		config_cur.BATCH_SIZE = 10 # GX:in the training, bs=8, then sp_bs+li_bs = 16.
		assert test_mode in ['test_A', 'test_B'], print("Please offer the valid mode.")
		dataset_test = Dataset(config_cur, test_mode+'_csv')
		# dataset_test = Dataset(config_cur, test_mode)
		img_num = len(dataset_test.name_list)
		num_list = int(img_num/config_cur.BATCH_SIZE)
		score0, score1, score2, score3, score4, label_lst = [],[],[],[],[],[]
		print(f"================================")
		print(f"working on the {prefix_dataset}.")
		for step in tqdm(range(num_list)):
			img, img_name = dataset_test.nextit()
			img_name = img_name.numpy().tolist()
			p_pretrain, dmap_pred_pretrain, region_map_pretrain, p, dmap_pred, region_map \
											= self._test_graph(img)
			p_pretrain, dmap_pred_pretrain, region_map_pretrain, p, dmap_pred, region_map = \
						p_pretrain.numpy(), dmap_pred_pretrain.numpy(), region_map_pretrain.numpy(), \
						p.numpy(), dmap_pred.numpy(), region_map.numpy()
			for i in range(config_cur.BATCH_SIZE):
				img_name_cur = img_name[i].decode('UTF-8')
				if ("Live" in img_name_cur) or ("live" in img_name_cur):
					label_cur = 0
				else:
					label_cur = 1
				label_lst.append(label_cur)
				self.csv_writer.writerow([img_name_cur, prefix_dataset, 
										  np.mean(p_pretrain[i]),
										  np.mean(dmap_pred_pretrain[i]),
										  np.mean(region_map_pretrain[i]), 
										  np.mean(p[i]), 
										  np.mean(dmap_pred[i]),
										  np.mean(region_map[i]),
										  label_cur, test_mode])
		config_cur.BATCH_SIZE = old_bs
		return score0, score1, score2, score3, label_lst

	@tf.function
	def _test_graph(self, img):
		dmap_pred_pretrain, p_pretrain, _, _, x_pretrain, region_map_pretrain = \
											  self.gen_pretrained(img, training=False)
		dmap_pred, p, c, n, x, region_map = self.gen(img, training=False)
		
		p_pretrain = tf.reduce_mean(p_pretrain, axis=[1,2,3])
		dmap_pred_pretrain = tf.reduce_mean(dmap_pred_pretrain, axis=[1,2,3])
		p = tf.reduce_mean(p, axis=[1,2,3])
		dmap_pred = tf.reduce_mean(dmap_pred, axis=[1,2,3])

		return p_pretrain, dmap_pred_pretrain, region_map_pretrain, p, dmap_pred, region_map

	def test_update(self, label_list, final_score2list, epoch, test_mode):
		APCER, BPCER, ACER, EER, res_tpr_05, auc_score, [tpr_fpr_h, tpr_fpr_m, tpr_fpr_l] \
						= my_metrics(label_list, final_score2list, val_phase=False)
		# print(len(label_list))
		message_cur = f"Test: ACER is {ACER:.3f}, EER is {EER:.3f}, AUC is {auc_score:.3f},  " 
		message_cur += f"tpr_fpr_03 is {tpr_fpr_m:.3f} and tpr_fpr_01 is {tpr_fpr_l:.3f}"
		self.log.display_metric(message_cur)
		self.SUMMARY_WRITER.add_scalar(f'{test_mode}/APCER', APCER, epoch)
		self.SUMMARY_WRITER.add_scalar(f'{test_mode}/BPCER', BPCER, epoch)
		self.SUMMARY_WRITER.add_scalar(f'{test_mode}/ACER',  ACER, epoch)
		self.SUMMARY_WRITER.add_scalar(f'{test_mode}/EER',   EER, epoch)
		self.SUMMARY_WRITER.add_scalar(f'{test_mode}/AUC',   auc_score, epoch)
		self.SUMMARY_WRITER.add_scalar(f'{test_mode}/tpr_fnr_05', res_tpr_05, epoch)
		self.SUMMARY_WRITER.add_scalar(f'{test_mode}/tpr_fpr_05', tpr_fpr_h, epoch)
		self.SUMMARY_WRITER.add_scalar(f'{test_mode}/tpr_fpr_03', tpr_fpr_m, epoch)
		self.SUMMARY_WRITER.add_scalar(f'{test_mode}/tpr_fpr_01', tpr_fpr_l, epoch)

def main(args):

	# Base Configuration Class
	config = Config(args)
	config_siw  = Config_siw(args)
	config_oulu = Config_oulu(args)

	config.lr   = args.lr
	config.type = args.type
	config.pretrain_folder = args.pretrain_folder
	config.desc_str = '_data_'+args.data+'_stage_'+config.phase+\
					  '_type_'+config.type+\
					  '_decay_'+str(config.DECAY_STEP)+\
					  '_epoch_'+str(args.epoch)+'_lr_'+\
					  str(config.lr)+'_spoof_region_architecture'
	config.root_dir = './log'+config.desc_str
	config.exp_dir  = '/exp'+config.desc_str
	config.CHECKPOINT_DIR = config.root_dir+config.exp_dir
	config.tb_dir   = './tb_logs'+config.desc_str
	config.save_model_dir = "./save_model"+config.desc_str
	config.SUMMARY_WRITER = SummaryWriter(config.tb_dir)
	config.csv_file_name  = "./"+config.desc_str+'.csv'
	config.epoch_num_pick = args.epoch_num
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
	config.compile_siwm()
	config_siw.compile()
	config_oulu.compile()
	print(config.BATCH_SIZE + config_siw.BATCH_SIZE + config_oulu.BATCH_SIZE)
	print('**********************************************************')

	stdnet = STDNet(config, config_siw, config_oulu, args)
	dataset_train_siwm = Dataset(config, 'train')
	dataset_train_siw  = Dataset(config_siw, 'train')
	dataset_train_oulu = Dataset(config_oulu, 'train')
	stdnet.train(dataset_train_siwm, dataset_train_siw, dataset_train_oulu, config)
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--cuda', type=int, default=3, help='The gpu num to use.')
  parser.add_argument('--stage', type=str, default='ft', choices=['ft'])
  parser.add_argument('--type', type=str, default='spoof', choices=['spoof','age','race','illu'])
  parser.add_argument('--set', type=str, default='all', help='To choose from the predefined 14 types.')
  parser.add_argument('--epoch', type=int, default=60, help='How many epochs to train the model.')
  parser.add_argument('--data', type=str, default='all', choices=['all','SiW','SiWM','oulu'])
  parser.add_argument('--lr', type=float, default=1e-7, help='The starting learning rate.')
  parser.add_argument('--decay_step', type=int, default=2, help='The learning rate decay step.')
  parser.add_argument('--pretrain_folder', type=str, default='./', help='Deprecated function.')
  parser.add_argument('--epoch_num', type=int, default=59, help='To pick the best epoch num.')
  parser.add_argument('--debug_mode', type=str, default='False', 
  									  choices=['True', "False"], help='Deprecated function.')
  args = parser.parse_args()
  main(args)
