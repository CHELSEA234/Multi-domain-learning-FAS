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
import numpy as np
from model import Generator, Discriminator, region_estimator
from utils import l1_loss, l2_loss, hinge_loss, Logging
from dataset import Dataset
from config import Config_siwm as Config
from config import Config_siw, Config_oulu
from metrics import my_metrics
from tensorboardX import SummaryWriter


class STDNet(object):
	def __init__(self, config, config_siw, config_oulu):
		self.config = config
		self.config_siw  = config_siw
		self.config_oulu = config_oulu
		self.lr = config.lr
		self.bs = config.BATCH_SIZE + config_siw.BATCH_SIZE + config_oulu.BATCH_SIZE
		self.SUMMARY_WRITER = config.SUMMARY_WRITER
		
		## The pretrained set:
		self.gen_pretrained  = Generator()

		## The new set:
		self.gen = Generator()
		self.disc1 = Discriminator(1,config.n_layer_D)
		self.disc2 = Discriminator(2,config.n_layer_D)
		self.disc3 = Discriminator(4,config.n_layer_D) 
		self.gen_opt = tf.keras.optimizers.Adam(self.lr)
		self.disc_opt = tf.keras.optimizers.Adam(self.lr)

		# Checkpoint initialization.
		self.save_dir = config.save_model_dir
		self.checkpoint_path_g = self.save_dir+"/gen/cp-{epoch:04d}.ckpt"
		self.checkpoint_path_d1= self.save_dir+"/dis1/cp-{epoch:04d}.ckpt"
		self.checkpoint_path_d2= self.save_dir+"/dis2/cp-{epoch:04d}.ckpt"
		self.checkpoint_path_d3= self.save_dir+"/dis3/cp-{epoch:04d}.ckpt"
		self.checkpoint_path_g_op = self.save_dir+"/g_opt/cp-{epoch:04d}.ckpt"
		self.checkpoint_path_d_op = self.save_dir+"/d_opt/cp-{epoch:04d}.ckpt"
		
		self.checkpoint_dir_g    = os.path.dirname(self.checkpoint_path_g)
		self.checkpoint_dir_d1   = os.path.dirname(self.checkpoint_path_d1)
		self.checkpoint_dir_d2   = os.path.dirname(self.checkpoint_path_d2)
		self.checkpoint_dir_d3   = os.path.dirname(self.checkpoint_path_d3)
		self.checkpoint_dir_g_op = os.path.dirname(self.checkpoint_path_g_op)
		self.checkpoint_dir_d_op = os.path.dirname(self.checkpoint_path_d_op)

		self.model_list  = [self.gen, self.disc1, self.disc2, self.disc3]
		self.model_p_list= [self.checkpoint_path_g, 
							self.checkpoint_path_d1,
							self.checkpoint_path_d2,
							self.checkpoint_path_d3]
		self.model_d_list= [self.checkpoint_dir_g,
							self.checkpoint_dir_d1,
							self.checkpoint_dir_d2,
							self.checkpoint_dir_d3]

		# Log class for displaying the losses.
		self.log = Logging(config)
		self.gen_opt  = tf.keras.optimizers.Adam(self.lr)
		self.disc_opt = tf.keras.optimizers.Adam(self.lr)

		# save the files:
		self.csv_file = None

	#############################################################################
	def inference(self, config):
		# GX: so far, you never break and resume the code...
		self.csv_file = open(self.config.csv_file_name, mode='w')
		self.csv_writer = csv.writer(self.csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		self.csv_writer.writerow(['Video name', 'Dataset', 'Score', 'Label', 'test_mode'])

		epoch_num = self.config.epoch_num_pick
		self.log.epoch = epoch_num
		for model_, model_dir_ in zip(self.model_list, self.model_d_list):
			epoch_suffix = f"/cp-{epoch_num:04d}.ckpt"
			current_checkpoint = model_dir_ + epoch_suffix
			model_.load_weights(current_checkpoint)

		start = time.time()
		for test_mode in ["test_A", "test_B"]:
			_, final_score2list, _, _, label_list = self.test_step(test_mode)
			assert len(label_list) == len(final_score2list), print("Their length should match.")
			print ('\n*****Time for epoch {} is {} sec*****'.format(epoch_num + 1, 
																	int(time.time()-start)))
		self.csv_file.close()
		self.SUMMARY_WRITER.close()

	def test_step(self, test_mode, viz_mode=False):
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
		return final_score0list, final_score2list, final_score3list, final_score4list, label_list  

	def test_step_helper(self, config_cur, test_mode, viz_mode=False, prefix_dataset=None):
		old_bs = config_cur.BATCH_SIZE
		config_cur.BATCH_SIZE = 16 
		dataset_test = Dataset(config_cur, test_mode+'_csv')

		img_num = len(dataset_test.name_list)
		num_list = int(img_num/config_cur.BATCH_SIZE)
		score0, score1, score2, score3, score4, label_lst = [],[],[],[],[],[]
		for step in range(num_list):
			img, img_name = dataset_test.nextit()
			img_name = img_name.numpy().tolist()
			p, p_area = self._test_graph(img)
			p = p.numpy()
			final_score2 = p/2
			score2.extend(final_score2)
			for i in range(config_cur.BATCH_SIZE):
				img_name_cur = img_name[i].decode('UTF-8')
				if ("Live" in img_name_cur) or ("live" in img_name_cur):
					label_cur = 0
				else:
					label_cur = 1
				label_lst.append(label_cur)
				self.csv_writer.writerow([img_name_cur, prefix_dataset, final_score2[i], label_cur, test_mode])
			self.log.step = step
		config_cur.BATCH_SIZE = old_bs
		return score0, score2, score3, score4, label_lst

	@tf.function
	def _test_graph(self, img):
		dmap_pred, p_area, c, n, x, region_map = self.gen(img, training=False)
		p = tf.reduce_mean(p_area, axis=[1,2,3])
		return p, p_area

	def test_update(self, label_list, final_score2list, epoch, test_mode):
		APCER, BPCER, ACER, EER, res_tpr_05, auc_score, [tpr_fpr_h, tpr_fpr_m, tpr_fpr_l] \
						= my_metrics(label_list, final_score2list, val_phase=False)
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
					  str(config.lr)
	config.root_dir = './log'+config.desc_str
	config.exp_dir  = '/exp'+config.desc_str
	config.CHECKPOINT_DIR = config.root_dir+config.exp_dir
	config.tb_dir   = './tb_logs'+config.desc_str
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
	config.compile_siwm()
	config_siw.compile()
	config_oulu.compile()
	print('**********************************************************')

	stdnet = STDNet(config, config_siw, config_oulu)
	stdnet.inference(config)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--cuda', type=int, default=3, help='The gpu num to use.')
  parser.add_argument('--stage', type=str, default='ft', choices=['ft','pretrain','ub'])
  parser.add_argument('--type', type=str, default='spoof', choices=['spoof','age','race','illu'])
  parser.add_argument('--set', type=str, default='all', help='To choose from the predefined 14 types.')
  parser.add_argument('--epoch', type=int, default=60, help='How many epochs to train the model.')
  parser.add_argument('--data', type=str, default='all', choices=['all','SiW','SiWM','oulu'])
  parser.add_argument('--lr', type=float, default=1e-5, help='The starting learning rate.')
  parser.add_argument('--decay_step', type=int, default=2, help='The learning rate decay step.')
  parser.add_argument('--pretrain_folder', type=str, default='./', help='Deprecated function.')
  args = parser.parse_args()
  main(args)