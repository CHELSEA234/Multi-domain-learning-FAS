from metrics_ import my_metrics
import tensorflow as tf
import argparse
import os
import time
import math
import numpy as np
from model import Generator, Discriminator, region_estimator, Multi_Con_Discriminator, Feature_transform_layer
from dataset_full import Dataset
from utils import l1_loss, l2_loss, hinge_loss, Logging
from config_full import Config_siwm as Config
from config_full import Config_siw, Config_oulu
from tensorboardX import SummaryWriter
from save_model_path import *

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
		print(checkpoint_dir)
		last_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
		# print(last_checkpoint)
		# print(last_checkpoint.split('/'))

		if pretrain:
			last_checkpoint_lst = last_checkpoint.split('/')
			# last_checkpoint[-1] = "cp-0040.ckpt"
			last_checkpoint = "/".join(last_checkpoint_lst[:-1]) + "/cp-0040.ckpt"
		print(last_checkpoint)
		# last_checkpoint = last_checkpoint.replace("cp-0059.ckpt","cp-0040.ckpt")
		# import sys;sys.exit(0)
		model.load_weights(last_checkpoint)
		if not pretrain:
			last_epoch = int((last_checkpoint.split('.')[1]).split('-')[-1])
			return last_epoch

	def _save(self, model, checkpoint_path, epoch):
		model.save_weights(checkpoint_path.format(epoch=epoch))

	#############################################################################
	def train(self, dataset, dataset_siw, dataset_oulu, config):

		# for i, j in zip(self.model_list, self.model_d_list):
			# last_epoch = self._restore(i, j)
		# last_epoch = 1
		# self.gen.load_weights("./save_model_data_all_stage_ft_type_spoof_decay_1_epoch_60_lr_1e-07_spoof_region_architecture/gen/cp-0001.ckpt")
		# self.RE.load_weights("./save_model_data_all_stage_ft_type_spoof_decay_1_epoch_60_lr_1e-07_spoof_region_architecture/ReE/cp-0001.ckpt")
		# self.disc1.load_weights("./save_model_data_all_stage_ft_type_spoof_decay_1_epoch_60_lr_1e-07_spoof_region_architecture/dis1/cp-0001.ckpt")
		# self.disc2.load_weights("./save_model_data_all_stage_ft_type_spoof_decay_1_epoch_60_lr_1e-07_spoof_region_architecture/dis2/cp-0001.ckpt")
		# self.disc3.load_weights("./save_model_data_all_stage_ft_type_spoof_decay_1_epoch_60_lr_1e-07_spoof_region_architecture/dis3/cp-0001.ckpt")
		# self.layer_1.load_weights("./save_model_data_all_stage_ft_type_spoof_decay_1_epoch_60_lr_1e-07_spoof_region_architecture/l1/cp-0001.ckpt")
		# self.layer_2.load_weights("./save_model_data_all_stage_ft_type_spoof_decay_1_epoch_60_lr_1e-07_spoof_region_architecture/l2/cp-0001.ckpt")
		# self.disc_kd.load_weights("./save_model_data_all_stage_ft_type_spoof_decay_1_epoch_60_lr_1e-07_spoof_region_architecture/kdd/cp-0001.ckpt")
		# print('**********************************************************')
		# print('Restore from Epoch '+str(last_epoch))
		# self.update_lr(restore=True, last_epoch=last_epoch)
		# print('**********************************************************')
		# self._restore(self.gen_pretrained, pretrain_model_dir, True)
		# print('**********************************************************')
		# print("finish loading the pretrain-model.")
		# print(pretrain_model_dir)

		last_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir_g)
		# print(last_checkpoint)
		# print("...over...")
		# import sys;sys.exit(0)
		if last_checkpoint:
			# print("coming here...")
			# print("coming here...")
			# import sys;sys.exit(0)
			for i, j in zip(self.model_list, self.model_d_list):
				last_epoch = self._restore(i, j)
			print('**********************************************************')
			print('Restore from Epoch '+str(last_epoch))
			self.update_lr(restore=True, last_epoch=last_epoch)
			print('**********************************************************')
		else:
			self._restore(self.gen_pretrained, pretrain_model_dir, True)
			print('**********************************************************')
			print("finish loading the pretrain-model.")
			print(pretrain_model_dir)
			print('**********************************************************')
			pretrain_model_dir_list = [pretrain_RE_model_dir+'ReE',
									   pretrain_RE_model_dir+'gen',
									   pretrain_RE_model_dir+'dis1',
									   pretrain_RE_model_dir+'dis2',
									   pretrain_RE_model_dir+'dis3']
			for i, j in zip([self.gen, self.RE, self.disc1, self.disc2, self.disc3], 
							 pretrain_model_dir_list):
				last_epoch = self._restore(i, j, True)
			print('**********************************************************')
			print("finish loading the pretrain-model.")
			print(pretrain_RE_model_dir)
			print('**********************************************************')
			last_epoch = 0

		for epoch in range(last_epoch, self.config.MAX_EPOCH):
			start = time.time()
			training = True
			for step in range(self.config.STEPS_PER_EPOCH):
				img_batch0 = dataset.nextit()
				img_batch1 = dataset_siw.nextit()
				img_batch2 = dataset_oulu.nextit()
				losses, figs = self.train_step(img_batch0, img_batch1, img_batch2, 
											   training, tf.constant(step))
				
				# display message every TXT_LOG_FR steps; save fig every IMG_LOG_FR steps.
				self.log.display(losses, epoch, step, training, self.config.STEPS_PER_EPOCH)
				self.log.save(figs, training)	# config.IMG_LOG_FR
				iter_num = self.config.STEPS_PER_EPOCH*epoch+step
				for name_, loss_ in losses.items():
					self.SUMMARY_WRITER.add_scalar(f'train/{name_}', loss_.numpy(), iter_num)
				# if step == 10:
				# 	import sys;sys.exit(0)

			for i, j in zip(self.model_list, self.model_p_list):
				self._save(i, j, epoch)
			
			if epoch % config.DECAY_STEP == 0:
				self.SUMMARY_WRITER.add_scalar(f'train/gen_lr', 
												self.gen_opt.learning_rate.numpy(), epoch)
				self.SUMMARY_WRITER.add_scalar(f'train/disc_lr', 
												self.disc_opt.learning_rate.numpy(), epoch)
				self.lr = self.lr * config.LEARNING_RATE_DECAY_FACTOR
				self.update_lr(self.lr)
			self.SUMMARY_WRITER.flush()

		self.SUMMARY_WRITER.close()

	@tf.function
	def correlation_matrix(self, region_map):
		region_map_ = tf.image.resize(region_map, [8, 8])
		b, w, h, _ = region_map_.shape
		# print(b,w,h)
		tsne_feature = tf.reshape(region_map_, [b, w*h])
		# print("shape is: ", tsne_feature.get_shape())
		tsne_featureT = tf.transpose(tsne_feature)
		# print("shape is: ", tsne_featureT.get_shape())
		# import sys;sys.exit(0)
		matrix_1   = tf.matmul(tsne_feature, tsne_featureT)
		# print(tsne_feature.get_shape())
		norm_value = tf.norm(tsne_feature+1e-3, axis=1)
		# print(norm_value.get_shape())
		norm_value = tf.reshape(norm_value, [-1, 1])
		norm_valueT = tf.transpose(norm_value)
		matrix_2 = tf.matmul(norm_value, norm_valueT)
		## it has many Nan.
		res = matrix_1/matrix_2
		# res = matrix_2
		return res

	@tf.function
	def adversarial_loss(self, x, x_pretrain, scale=0.0001):
		d_kd_output = tf.constant([[1, 2, 3], [4, 5, 6]])
		for idx in range(4):
			new_fea = x[idx]
			pre_fea = x_pretrain[idx]
			## to define new_fea is 1 and pre_fea is 0.
			input_fea = tf.concat([new_fea, pre_fea], axis=0)
			# print(input_fea.get_shape())
			if idx == 0:
				input_fea = self.layer_1(input_fea)
			else:
				input_fea = self.layer_2(input_fea)
			d_kd_output = self.disc_kd(input_fea, d_kd_output, 
										first_input=(idx==0), 
										last_output=(idx==3))

		dis_kd_loss_new = l2_loss(tf.abs(d_kd_output[0]), 1)
		dis_kd_loss_pre = l2_loss(tf.abs(d_kd_output[1]), 0)
		gen_kd_loss = l2_loss(tf.abs(d_kd_output[0]), 0)
		kd_total_loss = (dis_kd_loss_new + dis_kd_loss_pre)*scale
		return kd_total_loss, gen_kd_loss, dis_kd_loss_new, dis_kd_loss_pre

	@tf.function
	def train_step(self, data, data1, data2, training, step=0):
		losses = {}
		figs = []
		# bsize  = self.config.BATCH_SIZE.
		# GX: this means self.bs real and self.bs fake. True BS = self.bs * 2
		bsize  = self.bs
		imsize = self.config.IMG_SIZE

		# Get images and labels for CNN.
		img_li0, img_sp0, dmap_li0, dmap_sp0, _, _, _ = data
		img_li1, img_sp1, dmap_li1, dmap_sp1, _, _, _ = data1
		img_li2, img_sp2, dmap_li2, dmap_sp2, _, _, _ = data2

		img_li = tf.concat([img_li0, img_li1, img_li2], axis=0)
		img_sp = tf.concat([img_sp0, img_sp1, img_sp2], axis=0)
		dmap_li = tf.concat([dmap_li0, dmap_li1, dmap_li2], axis=0)
		dmap_sp = tf.concat([dmap_sp0, dmap_sp1, dmap_sp2], axis=0)

		img = tf.concat([img_li, img_sp], axis=0)
		dmap = tf.concat([dmap_li, dmap_sp], axis=0)
		dmap_size_32 = tf.image.resize(dmap, [32, 32])
		bmask = tf.cast(tf.greater(dmap[..., 0], 0), tf.float32)

		###########################################################
		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, \
			 tf.GradientTape() as reg_tape, \
			 tf.GradientTape() as kd_tape,\
			 tf.GradientTape() as kg_tape:
			_, p_pretrain, _, _, x_pretrain, region_map_pretrain = self.gen_pretrained(img, training=training)
			dmap_pred, p, c, n, x, region_map = self.gen(img, training=training)

			# Live reconstruction.
			recon = (1 - p) * (img - n) + p * c
			trace = img - recon

			## region map:
			region_map_loss = l1_loss(region_map_pretrain, region_map)			

			## correlation mask:
			cor_matrix_pre = self.correlation_matrix(region_map_pretrain)
			cor_matrix 	   = self.correlation_matrix(region_map)
			corr_loss = l1_loss(cor_matrix_pre, cor_matrix)

			## adversarial training:
			kd_total_loss, gen_kd_loss, dis_kd_loss_new, dis_kd_loss_pre = self.adversarial_loss(x, x_pretrain)

			## New spoof synthesis.
			d_img = tf.concat([img[:bsize, ...], recon[bsize:, ...]], axis=0)  
			d_output_1 = self.disc1(d_img, training=training)
			d_output_2 = self.disc2(d_img, training=training)
			d_output_3 = self.disc3(d_img, training=training)

			## Inpainting mask loss.
			p_prior_knowledge = l1_loss(p[..., 0], dmap[..., 1])
			# p_significant_change = tf.stop_gradient(
			# 						tf.cast(
			# 							tf.greater(tf.reduce_sum(tf.abs(trace), axis=3), 0.3), 
			# 						tf.float32)
			# 					   )
			# A = tf.greater(tf.reduce_sum(tf.abs(trace[0:4,:,:]), axis=3), 0.95)
			# B = tf.greater(tf.reduce_sum(tf.abs(trace[4:6,:,:]), axis=3), 0.1)
			# C = tf.greater(tf.reduce_sum(tf.abs(trace[6:,:,:]), axis=3), 0.3)
			# ABC = tf.concat([A,B,C], axis=0)

			ABC = tf.greater(tf.reduce_sum(tf.abs(trace), axis=3), 0.3)
			# print(ABC.get_shape())
			p_significant_change = tf.stop_gradient(tf.cast(ABC, tf.float32))
			p_post_constraint = tf.reduce_mean(
									tf.abs(
										tf.squeeze(p[bsize:, ...])
								   		- p_significant_change[bsize:, ...]
								   		)
									)
			p_loss = p_prior_knowledge * 0.1 + p_post_constraint + region_map_loss + corr_loss
  	 	
  			# Trace constraint loss.
			trace_loss  = tf.reduce_mean(tf.abs(trace[:bsize, ...])) + \
						  tf.reduce_mean(tf.abs(trace[bsize:, ...])) * 1e-5

			# Depth map loss.
			dmap_loss = l1_loss(dmap_pred, dmap_size_32) * 100

			# GAN loss for the generator.
			gan_loss  = l2_loss(d_output_1[1], 1) + l2_loss(d_output_2[1], 1) + \
						l2_loss(d_output_3[1], 1)

			# Overall loss for generator.
			g_total_loss = dmap_loss + gan_loss + p_loss + trace_loss * 10 + gen_kd_loss

			# Discriminators loss.
			d_loss_r = l2_loss(d_output_1[0], 1) + l2_loss(d_output_2[0], 1) + \
					   l2_loss(d_output_3[0], 1)
			d_loss_s = l2_loss(d_output_1[1], 0) + l2_loss(d_output_2[1], 0) + \
					   l2_loss(d_output_3[1], 0)
			d_total_loss =  (d_loss_r + d_loss_s) / 4
		
		if training:
			# Gather all the trainable variables
			gen_trainable_vars  = self.gen.trainable_variables
			disc_trainable_vars = self.disc1.trainable_variables + \
								  self.disc2.trainable_variables + \
								  self.disc3.trainable_variables
			kdis_trainable_vars = self.disc_kd.trainable_variables
			kgen_trainable_vars = self.layer_1.trainable_variables + \
								  self.layer_2.trainable_variables
			# Generate gradients.
			g_gradients = gen_tape.gradient(g_total_loss, gen_trainable_vars)
			d_gradients = disc_tape.gradient(d_total_loss, disc_trainable_vars)
			kg_gradients= kg_tape.gradient(gen_kd_loss, kgen_trainable_vars)
			kd_gradients= kd_tape.gradient(kd_total_loss, kdis_trainable_vars)
			# Backpropogate gradients.
			self.gen_opt.apply_gradients(zip(g_gradients, gen_trainable_vars))
			self.gen_opt.apply_gradients(zip(kg_gradients, kgen_trainable_vars))
			if step % 2 == 0:
				self.disc_opt.apply_gradients(zip(d_gradients, disc_trainable_vars))
			if step % 10 == 0:
				self.disc_opt.apply_gradients(zip(kd_gradients, kdis_trainable_vars))
	    # Gather losses for displaying for tracking the training. 
	    # Not using Tensorboard here but you can write your own.
		losses['dmap'] = dmap_loss
		losses['gen']  = gan_loss
		losses['kd_d_new'] = dis_kd_loss_new
		losses['kd_d_pre'] = dis_kd_loss_pre
		losses['region_map_loss'] = region_map_loss
		losses['corr_loss'] = corr_loss
		losses['p_post_constraint'] = p_post_constraint
		losses['disc_real'] = d_loss_r
		losses['disc_fake'] = d_loss_s
		losses['p_prior_knowledge'] = p_prior_knowledge * 0.1
		losses['trace_loss'] = trace_loss
		# Gather network output and intermediate results for visualization.
		dmap = tf.concat([dmap, tf.zeros([bsize*2, 256, 256, 1])], axis=3)
		dmap_pred = tf.concat([dmap_pred, tf.zeros([bsize*2, 32, 32, 1])], axis=3)
		# region_map = tf.reshape(region_map, [2*bsize,256,256,1])
		figs = [img, recon, tf.abs(p), tf.abs(region_map), tf.abs(region_map_pretrain), 
				tf.abs(c), tf.abs(n)]
		return losses, figs

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
  parser.add_argument('--debug_mode', type=str, default='False', 
  									  choices=['True', "False"], help='Deprecated function.')
  args = parser.parse_args()
  main(args)
