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
import numpy as np
from model import Generator, Discriminator, region_estimator
from utils import l1_loss, l2_loss, Logging
from dataset import Dataset
from config_siwm import Config_siwm
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
		self.log = Logging(config)

	def update_lr(self, new_lr=0, restore=False, last_epoch=0):
		if restore:
			assert last_epoch != 0, print("Restoring LR should not start at 0 epoch.")
			self.lr = self.lr * np.power(self.config.LEARNING_RATE_DECAY_FACTOR, last_epoch)
			print(f"Restoring the previous learning rate {self.lr} at epoch {last_epoch}.")
		self.gen_opt.learning_rate.assign(self.lr)

	def _restore(self, model, checkpoint_dir, pretrain=False):
		if not pretrain:
			last_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
			model.load_weights(last_checkpoint)
			last_epoch = int((last_checkpoint.split('.')[1]).split('-')[-1])
			return last_epoch
		else:
			model.load_weights(checkpoint_dir+'/cp-0179.ckpt')

	def _save(self, model, checkpoint_path, epoch):
		model.save_weights(checkpoint_path.format(epoch=epoch))

	#############################################################################
	def train(self, dataset, config):
		last_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir_g)
		if last_checkpoint:
			for i, j in zip(self.model_list, self.model_d_list):
				last_epoch = self._restore(i, j)
			print('**********************************************************')
			print('Restore from Epoch '+str(last_epoch))
			self.update_lr(restore=True, last_epoch=last_epoch)
			print('**********************************************************')
		else:
			print('**********************************************************')
			print('Training from the scratch.')
			print('**********************************************************')
			last_epoch = 0

		for epoch in range(last_epoch, self.config.MAX_EPOCH):
			start = time.time()
			training = True
			for step in range(self.config.STEPS_PER_EPOCH):
				img_batch = dataset.nextit()
				losses, figs = self.train_step(img_batch, training, tf.constant(step))
				self.log.display(losses, epoch, step, training, self.config.STEPS_PER_EPOCH)
				self.log.save(figs, training)
				iter_num = self.config.STEPS_PER_EPOCH*epoch+step
				for name_, loss_ in losses.items():
					self.SUMMARY_WRITER.add_scalar(f'train/{name_}', loss_.numpy(), iter_num)
			for i, j in zip(self.model_list, self.model_p_list):
				self._save(i, j, epoch)
			if epoch % config.DECAY_STEP == 0:
				self.SUMMARY_WRITER.add_scalar(f'train/gen_lr', self.gen_opt.learning_rate.numpy(), epoch)
				self.lr = self.lr * config.LEARNING_RATE_DECAY_FACTOR
				self.update_lr(self.lr)
			self.SUMMARY_WRITER.flush()
		self.SUMMARY_WRITER.close()

	@tf.function
	def train_step(self, data, training, step=0):
		losses = {}
		figs = []
		bsize, imsize  = self.bs, self.config.IMG_SIZE
		img_li, img_sp, dmap_li, dmap_sp, _, _, _ = data
		img = tf.concat([img_li, img_sp], axis=0)
		dmap = tf.concat([dmap_li, dmap_sp], axis=0)
		dmap_size_32 = tf.image.resize(dmap, [32, 32])

		###########################################################
		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as reg_tape:
			dmap_pred, p, c, n, x, region_map = self.gen(img, training=training)
			region_map = tf.reshape(region_map, [2*bsize,256,256,1])

			# Live reconstruction.
			recon = (1 - p) * (img - n) + p * c
			trace = img - recon
			
			d_img = tf.concat([img[:bsize, ...], recon[bsize:, ...]], axis=0)  
			d_output_1 = self.disc1(d_img, training=training)
			d_output_2 = self.disc2(d_img, training=training)
			d_output_3 = self.disc3(d_img, training=training)

			# Semantic mask loss.
			p_prior_knowledge = l1_loss(p[..., 0], dmap[..., 1])
			real_change = tf.zeros([4,256,256])
			siwm_change = tf.cast(tf.greater(tf.reduce_sum(tf.abs(trace[4:,:,:]), axis=3), 0.35),tf.float32)
			p_significant_change = tf.stop_gradient(tf.concat([real_change, siwm_change], axis=0))
			map_loss = l1_loss(tf.squeeze(region_map), p_significant_change)
			p_post_constraint = tf.abs(tf.squeeze(p[bsize:, ...]) - p_significant_change[bsize:, ...])
			p_post_constraint = tf.reduce_mean(p_post_constraint)
			p_loss = p_prior_knowledge * 0.1 + p_post_constraint
  		
  			# Trace constraint loss.
			trace_loss  = tf.reduce_mean(tf.abs(trace[:bsize, ...])) + \
						  tf.reduce_mean(tf.abs(trace[bsize:, ...])) * 1e-5

			# Depth map loss.
			dmap_loss = l1_loss(dmap_pred, dmap_size_32) * 100

			# GAN loss for the generator.
			gan_loss  = l2_loss(d_output_1[1], 1) + l2_loss(d_output_2[1], 1) + l2_loss(d_output_3[1], 1)

			# Overall loss for generator.
			g_total_loss = dmap_loss + gan_loss + p_loss + trace_loss * 10

			# Discriminators loss.
			d_loss_r = l2_loss(d_output_1[0], 1) + l2_loss(d_output_2[0], 1) + l2_loss(d_output_3[0], 1)
			d_loss_s = l2_loss(d_output_1[1], 0) + l2_loss(d_output_2[1], 0) + l2_loss(d_output_3[1], 0)
			d_total_loss =  (d_loss_r + d_loss_s) / 4
		
		if training:
			# Gather all the trainable variables
			gen_trainable_vars  = self.gen.trainable_variables
			reg_trainable_vars  = self.RE.trainable_variables
			disc_trainable_vars = self.disc1.trainable_variables + \
								  self.disc2.trainable_variables + \
								  self.disc3.trainable_variables

			# Generate gradients.
			r_gradients = reg_tape.gradient(map_loss, reg_trainable_vars)
			g_gradients = gen_tape.gradient(g_total_loss, gen_trainable_vars)
			d_gradients = disc_tape.gradient(d_total_loss, disc_trainable_vars)

			# Backpropogate gradients.
			self.gen_opt.apply_gradients(zip(g_gradients, gen_trainable_vars))
			self.gen_opt.apply_gradients(zip(r_gradients, reg_trainable_vars))
			if step % 2 == 0:
				self.gen_opt.apply_gradients(zip(d_gradients, disc_trainable_vars))

	    # Gather losses for displaying for tracking the training. 
		losses['dmap'] = dmap_loss
		losses['gen']  = gan_loss
		losses['map_loss'] = map_loss
		losses['p_post_constraint'] = p_post_constraint
		losses['disc_real'] = d_loss_r
		losses['disc_fake'] = d_loss_s
		losses['p_prior_knowledge'] = p_prior_knowledge * 0.1
		losses['trace_loss'] = trace_loss
		# Gather network output and intermediate results for visualization.
		dmap = tf.concat([dmap, tf.zeros([bsize*2, 256, 256, 1])], axis=3)
		dmap_pred = tf.concat([dmap_pred, tf.zeros([bsize*2, 32, 32, 1])], axis=3)
		p_significant_change = tf.expand_dims(p_significant_change, axis=-1)
		figs = [img, recon, tf.abs(p), tf.abs(p_significant_change), 
				tf.abs(region_map), tf.abs(c), tf.abs(n), dmap, dmap_pred]
		return losses, figs

def main(args):
	# Base Configuration Class
	config = Config_siwm(args)
	config.lr   = args.lr
	config.type = args.type
	config.DECAY_STEP = args.decay_step
	config.pretrain_folder = args.pretrain_folder
	config.desc_str = '_data_'+args.data+\
					  '_protocol_'+str(config.protocol)+\
					  '_unknown_'+config.unknown+\
					  '_decay_'+str(config.DECAY_STEP)+\
					  '_epoch_'+str(args.epoch)+\
					  '_lr_'+str(config.lr)+'_siw_mv2'
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
	config.compile()

	print('**********************************************************')
	srenet  = SRENet(config)
	dataset_train = Dataset(config, 'train')
	srenet.train(dataset_train, config)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--cuda', type=int, default=6, help='The gpu num to use.')
  parser.add_argument('--stage', type=str, default='ft', choices=['ft','pretrain','ub'])
  parser.add_argument('--type', type=str, default='spoof', choices=['spoof','age','race','illu'])
  parser.add_argument('--set', type=str, default='all', help='To choose from the predefined 14 types.')
  parser.add_argument('--epoch', type=int, default=180, help='How many epochs to train the model.')
  parser.add_argument('--data', type=str, default='all', choices=['all','SiW','SiWM','oulu'])
  parser.add_argument('--lr', type=float, default=1e-4, help='The starting learning rate.')
  parser.add_argument('--decay_step', type=int, default=3, help='The learning rate decay step.')
  parser.add_argument('--pretrain_folder', type=str, default='./pre_trained/', help='Pretrain weight.')
  parser.add_argument('--pro', type=int, default=1, help='Protocol number.')
  parser.add_argument('--unknown', type=str, default='Ob', help='The unknown spoof type.')
  args = parser.parse_args()
  main(args)