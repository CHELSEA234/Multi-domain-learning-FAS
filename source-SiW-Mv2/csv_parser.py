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
from metrics import my_metrics
from glob import glob
import sys
import csv
import numpy as np
import argparse

def dump_to_screen(apcer_order, bpcer_order, acer_order, tpr_order,
					apcer, bpcer, acer, tpr):
	'''standard output results to screen.'''
	print('AP: ', apcer_order, f"MEAN: {np.mean(apcer)*100:.1f}", f"STD: {np.std(apcer)*100:.1f}")
	print('BP: ', bpcer_order, f"MEAN: {np.mean(bpcer)*100:.1f}", f"STD: {np.std(bpcer)*100:.1f}")
	print('ACER: ', acer_order, f"MEAN: {np.mean(acer)*100:.1f}", f"STD: {np.std(acer)*100:.1f}")
	print('TPR@FPR=1.0%: ', tpr_order, f"MEAN: {np.mean(tpr)*100:.1f}", f"STD: {np.std(tpr)*100:.1f}")
	print("...over...")
	sys.exit(0)

def compute_score(args, score_list, label_list, test_name, verbose=False):
	'''the depth + region performance.'''
	# print(test_name, len(score_list))
	APCER, BPCER, ACER, EER, res_tpr_05, auc_score, [tpr_fpr_h, tpr_fpr_m, tpr_fpr_l] \
			= my_metrics(label_list, score_list, val_phase=False)
	message_cur = f"Test: {args.weight:.1f} depth score \n"
	message_cur += f"ACER is {ACER*100:.1f}, AP: {APCER*100:.1f}, BP: {BPCER*100:.1f}, "
	message_cur += f"tpr_fpr_1.0% is {tpr_fpr_m*100:.1f}"
	if verbose:
		print(message_cur)
		print()
	return APCER, BPCER, ACER, tpr_fpr_m

def compute_metric(args, label_list, score_list, score2_list):
	'''parse the score dictionary here, based on the video-level evaluation.'''
	score_list, label_list = [], []
	live_sample_name, spoof_sample_name = [], []

	for key, value in score_dict.items():
		score_list_cur = score_dict[key]
		score_list_cur.sort()
		interval = int(len(score_list_cur)*args.interval)
		score_list_compute = score_list_cur[interval:][:-1-interval]
		if len(score_list_compute) == 0:
			continue
		# print(f'The key is: {key}, the score list length is {len(score_list_compute)}')
		value = np.mean(score_list_compute)
		score_list.append(np.mean(score_list_compute))
		if np.mean(score_list_compute) > 0.1:
			print(key)
		if "Live_" in key:
			label_list.append(0)
			live_sample_name.append(key)
		else:
			label_list.append(1)
			spoof_sample_name.append(key)
	assert len(label_list) == len(score_list)
	print(f'the total sample number is: {len(label_list)}.')
	print(f"the live sample and spoof numbers are: {len(live_sample_name)}, {len(spoof_sample_name)}.")
	return score_list, label_list

def video_level_compute(score_dict):
	'''computing the video level results.'''
	score_list, label_list = [], []
	live_sample_name, spoof_sample_name = [], []
	for key, value in score_dict.items():
		score_list_cur = score_dict[key]
		score_list_cur.sort()
		interval = int(len(score_list_cur)*args.interval)
		score_list_compute = score_list_cur[interval:][:-1-interval]
		if len(score_list_compute) == 0:
			continue
		# print(f'The key is: {key}, the score list length is {len(score_list_compute)}')
		value = np.mean(score_list_compute)
		score_list.append(np.mean(score_list_compute))
		if "Live_" in key:
			label_list.append(0)
			live_sample_name.append(key)
		else:
			label_list.append(1)
			spoof_sample_name.append(key)		
	return score_list, label_list

def return_dictionary(args, test_file_lst):
	'''
		return the score dictionary for different spoof types.
	'''
	for csv_file_name in test_file_lst:
		csv_file = open(csv_file_name)
		csv_reader = csv.reader(csv_file, delimiter=',')
		score_dict = dict() 	# key is video name, value is a list
		score_list, label_list, exist_sample_list = [], [], []
		score2_list = []
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				line_count += 1
			else:
				line_count += 1
				vid_name = row[0].split('/')[-2]
				label = float(row[-2])	
				depth_score, region_score = float(row[2]), float(row[3])
				final_score = depth_score + args.weight*region_score
				if vid_name not in score_dict:
					score_dict[vid_name] = [final_score]
				else:
					score_dict[vid_name].append(final_score)
				label_list.append(label)
				score_list.append(final_score)

		key_list = list(score_dict.keys())
		assert len(label_list) == len(score_list), print(len(label_list), len(score_list))
		
		if args.pro == 1:
			parse_protocol_1_result(score_dict)
		elif args.pro == 2:
			score_list, label_list = video_level_compute(score_dict)
			return score_list, label_list
		elif args.pro == 3:
			parse_protocol_3_result(score_dict)


def parse_protocol_1_result(score_dict):
	'''
		output result on different spoof attack on the protocol 1.
	'''
	print("Compute the protocol I scores.")
	co_list, eye_list, fun_eye_list, half_list, im_list, man_list, mouth_list, ob_list, \
	paper_list, paperglass_list, print_list, replay_list, sil_list, trans_list, live_list \
	= [],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
	spoof_type_dict = {
						'Co': 'Makeup_Co', 'Eye': 'Partial_Eye', 'Funnyeye': 'Partial_Funnyeye',
						'Half': 'Mask_Half', 'Im': 'Makeup_Im', 'Mann': 'Mask_Mann', 
						'Mouth': 'Partial_Mouth', 'Ob': 'Makeup_Ob', 'Paper': 'Mask_Paper', 
						'Paperglass': 'Partial_Paperglass', 'Print': 'Paper', 'Replay': 'Replay', 
						'Sil': 'Mask_Silicone', 'Trans': 'Mask_Trans'
						}
	vid_names = list(score_dict.keys())
	for _ in vid_names:
		if 'Live_' in _:
			live_list.append(np.mean(score_dict[_]))
		elif 'Makeup_Co' in _:
			co_list.append(np.mean(score_dict[_]))
		elif 'Partial_Eye' in _:
			eye_list.append(np.mean(score_dict[_]))
		elif 'Partial_Funnyeye' in _:
			fun_eye_list.append(np.mean(score_dict[_]))
		elif 'Mask_Half' in _:
			half_list.append(np.mean(score_dict[_]))
		elif 'Makeup_Im' in _:
			im_list.append(np.mean(score_dict[_]))
		elif 'Mask_Mann' in _:
			man_list.append(np.mean(score_dict[_]))
		elif 'Partial_Mouth' in _:
			mouth_list.append(np.mean(score_dict[_]))
		elif 'Makeup_Ob' in _:
			ob_list.append(np.mean(score_dict[_]))
		elif 'Mask_Paper' in _:
			paper_list.append(np.mean(score_dict[_]))
		elif 'Partial_Paperglass' in _:
			paperglass_list.append(np.mean(score_dict[_]))
		elif 'Paper' in _:
			print_list.append(np.mean(score_dict[_]))
		elif 'Replay' in _:
			replay_list.append(np.mean(score_dict[_]))
		elif 'Mask_Silicone' in _:
			sil_list.append(np.mean(score_dict[_]))
		elif 'Mask_Trans' in _:
			trans_list.append(np.mean(score_dict[_]))
		else:
			raise ValueError

	## GX: gathering the result.
	apcer, bpcer, acer, tpr = [], [], [], []
	apcer_order, bpcer_order, acer_order, tpr_order = [], [], [], []
	APCER, BPCER, ACER, tpr_fpr = compute_score(args, live_list + fun_eye_list, [0]*len(live_list) + [1]*len(fun_eye_list), test_name='fun_eye')	
	apcer.append(APCER);bpcer.append(BPCER);acer.append(ACER);tpr.append(tpr_fpr)
	apcer_order.append(f"{APCER*100:.1f}");bpcer_order.append(f"{BPCER*100:.1f}")
	acer_order.append(f"{ACER*100:.1f}");tpr_order.append(f"{tpr_fpr*100:.1f}")

	APCER, BPCER, ACER, tpr_fpr = compute_score(args, live_list + eye_list, [0]*len(live_list) + [1]*len(eye_list), test_name='eye')
	apcer.append(APCER);bpcer.append(BPCER);acer.append(ACER);tpr.append(tpr_fpr)
	apcer_order.append(f"{APCER*100:.1f}");bpcer_order.append(f"{BPCER*100:.1f}")
	acer_order.append(f"{ACER*100:.1f}");tpr_order.append(f"{tpr_fpr*100:.1f}")

	APCER, BPCER, ACER, tpr_fpr = compute_score(args, live_list + mouth_list, [0]*len(live_list) + [1]*len(mouth_list), test_name='mouth')
	apcer.append(APCER);bpcer.append(BPCER);acer.append(ACER);tpr.append(tpr_fpr)
	apcer_order.append(f"{APCER*100:.1f}");bpcer_order.append(f"{BPCER*100:.1f}")
	acer_order.append(f"{ACER*100:.1f}");tpr_order.append(f"{tpr_fpr*100:.1f}")

	APCER, BPCER, ACER, tpr_fpr = compute_score(args, live_list + paperglass_list, [0]*len(live_list) + [1]*len(paperglass_list), test_name='paperglass')
	apcer.append(APCER);bpcer.append(BPCER);acer.append(ACER);tpr.append(tpr_fpr)
	apcer_order.append(f"{APCER*100:.1f}");bpcer_order.append(f"{BPCER*100:.1f}")
	acer_order.append(f"{ACER*100:.1f}");tpr_order.append(f"{tpr_fpr*100:.1f}")

	APCER, BPCER, ACER, tpr_fpr = compute_score(args, live_list + im_list, [0]*len(live_list) + [1]*len(im_list), test_name='im')
	apcer.append(APCER);bpcer.append(BPCER);acer.append(ACER);tpr.append(tpr_fpr)
	apcer_order.append(f"{APCER*100:.1f}");bpcer_order.append(f"{BPCER*100:.1f}")
	acer_order.append(f"{ACER*100:.1f}");tpr_order.append(f"{tpr_fpr*100:.1f}")

	APCER, BPCER, ACER, tpr_fpr = compute_score(args, live_list + ob_list, [0]*len(live_list) + [1]*len(ob_list), test_name='ob')
	apcer.append(APCER);bpcer.append(BPCER);acer.append(ACER);tpr.append(tpr_fpr)
	apcer_order.append(f"{APCER*100:.1f}");bpcer_order.append(f"{BPCER*100:.1f}")
	acer_order.append(f"{ACER*100:.1f}");tpr_order.append(f"{tpr_fpr*100:.1f}")

	APCER, BPCER, ACER, tpr_fpr = compute_score(args, live_list + co_list, [0]*len(live_list) + [1]*len(co_list), test_name='co')
	apcer.append(APCER);bpcer.append(BPCER);acer.append(ACER);tpr.append(tpr_fpr)
	apcer_order.append(f"{APCER*100:.1f}");bpcer_order.append(f"{BPCER*100:.1f}")
	acer_order.append(f"{ACER*100:.1f}");tpr_order.append(f"{tpr_fpr*100:.1f}")

	APCER, BPCER, ACER, tpr_fpr = compute_score(args, live_list + half_list, [0]*len(live_list) + [1]*len(half_list), test_name='half')
	apcer.append(APCER);bpcer.append(BPCER);acer.append(ACER);tpr.append(tpr_fpr)
	apcer_order.append(f"{APCER*100:.1f}");bpcer_order.append(f"{BPCER*100:.1f}")
	acer_order.append(f"{ACER*100:.1f}");tpr_order.append(f"{tpr_fpr*100:.1f}")

	APCER, BPCER, ACER, tpr_fpr = compute_score(args, live_list + trans_list, [0]*len(live_list) + [1]*len(trans_list), test_name='trans')
	apcer.append(APCER);bpcer.append(BPCER);acer.append(ACER);tpr.append(tpr_fpr)
	apcer_order.append(f"{APCER*100:.1f}");bpcer_order.append(f"{BPCER*100:.1f}")
	acer_order.append(f"{ACER*100:.1f}");tpr_order.append(f"{tpr_fpr*100:.1f}")

	APCER, BPCER, ACER, tpr_fpr = compute_score(args, live_list + paper_list, [0]*len(live_list) + [1]*len(paper_list), test_name='paper')
	apcer.append(APCER);bpcer.append(BPCER);acer.append(ACER);tpr.append(tpr_fpr)
	apcer_order.append(f"{APCER*100:.1f}");bpcer_order.append(f"{BPCER*100:.1f}")
	acer_order.append(f"{ACER*100:.1f}");tpr_order.append(f"{tpr_fpr*100:.1f}")

	APCER, BPCER, ACER, tpr_fpr = compute_score(args, live_list + sil_list, [0]*len(live_list) + [1]*len(sil_list), test_name='sil')
	apcer.append(APCER);bpcer.append(BPCER);acer.append(ACER);tpr.append(tpr_fpr)
	apcer_order.append(f"{APCER*100:.1f}");bpcer_order.append(f"{BPCER*100:.1f}")
	acer_order.append(f"{ACER*100:.1f}");tpr_order.append(f"{tpr_fpr*100:.1f}")

	APCER, BPCER, ACER, tpr_fpr = compute_score(args, live_list + man_list, [0]*len(live_list) + [1]*len(man_list), test_name='man')
	apcer.append(APCER);bpcer.append(BPCER);acer.append(ACER);tpr.append(tpr_fpr)
	apcer_order.append(f"{APCER*100:.1f}");bpcer_order.append(f"{BPCER*100:.1f}")
	acer_order.append(f"{ACER*100:.1f}");tpr_order.append(f"{tpr_fpr*100:.1f}")

	APCER, BPCER, ACER, tpr_fpr = compute_score(args, live_list + replay_list, [0]*len(live_list) + [1]*len(replay_list), test_name='replay')
	apcer.append(APCER);bpcer.append(BPCER);acer.append(ACER);tpr.append(tpr_fpr)
	apcer_order.append(f"{APCER*100:.1f}");bpcer_order.append(f"{BPCER*100:.1f}")
	acer_order.append(f"{ACER*100:.1f}");tpr_order.append(f"{tpr_fpr*100:.1f}")

	APCER, BPCER, ACER, tpr_fpr = compute_score(args, live_list + print_list, [0]*len(live_list) + [1]*len(print_list), test_name='print')
	apcer.append(APCER);bpcer.append(BPCER);acer.append(ACER);tpr.append(tpr_fpr)
	apcer_order.append(f"{APCER*100:.1f}");bpcer_order.append(f"{BPCER*100:.1f}")
	acer_order.append(f"{ACER*100:.1f}");tpr_order.append(f"{tpr_fpr*100:.1f}")

	dump_to_screen(apcer_order, bpcer_order, acer_order, tpr_order, apcer, bpcer, acer, tpr)

def parse_protocol_2_result(spoof_lst, apcer, bpcer, acer, tpr):
	'''
		output results that can be easiliy copied to the overleaf.
	'''
	print("Compute the protocol II scores.")
	overleaf_order = ['Funnyeye', 'Eye', 'Mouth', 'Paperglass', 'Im', 'Ob', 'Co', 'Half', 'Trans', 
						'Paper', 'Sil', 'Mann', 'Replay', 'Print'
						]
	apcer_order, bpcer_order, acer_order, tpr_order = [], [], [], []
	for _ in overleaf_order:
		spoof_idx = spoof_lst.index(_)
		apcer_order.append(f"{apcer[spoof_idx]*100:.1f}")
		bpcer_order.append(f"{bpcer[spoof_idx]*100:.1f}")
		acer_order.append(f"{acer[spoof_idx]*100:.1f}")
		tpr_order.append(f"{tpr[spoof_idx]*100:.1f}")

	dump_to_screen(apcer_order, bpcer_order, acer_order, tpr_order, apcer, bpcer, acer, tpr)

def parse_protocol_3_result(score_dict):
	'''
		output results on protocol 3.
	'''
	print("Compute the protocol III scores.")
	apcer, bpcer, acer, tpr = [], [], [], []
	apcer_order, bpcer_order, acer_order, tpr_order = [], [], [], []
	target_file_list = [
						'./pro_3_text/test_A_pretrain.txt', 
						'./pro_3_text/test_B_spoof.txt',
						'./pro_3_text/test_C_race.txt', 
						'./pro_3_text/test_D_age.txt',
						'./pro_3_text/test_E_ill.txt'
						]
	for target_file_name in target_file_list:
		f = open(target_file_name, 'r')
		lines = f.readlines()
		sub_live_list, sub_spoof_list = [], []
		for line in lines:
			line = line.strip()
			if 'Live_' in line:
				sub_live_list.append(np.mean(score_dict[line]))
			else:
				try:
					sub_spoof_list.append(np.mean(score_dict[line]))
				except:
					continue
		APCER, BPCER, ACER, tpr_fpr = compute_score(args, sub_live_list+sub_spoof_list, 
													[0]*len(sub_live_list)+[1]*len(sub_spoof_list), 
													target_file_name, False)
		f.close()
		apcer.append(APCER);bpcer.append(BPCER);acer.append(ACER);tpr.append(tpr_fpr)
		apcer_order.append(f"{APCER*100:.1f}");bpcer_order.append(f"{BPCER*100:.1f}")
		acer_order.append(f"{ACER*100:.1f}");tpr_order.append(f"{tpr_fpr*100:.1f}")

	dump_to_screen(apcer_order, bpcer_order, acer_order, tpr_order, apcer, bpcer, acer, tpr)

def main(args):
	## The performance is evaluated on the video-level.
	folder_list = glob(f'{args.log_dir}/*/*.csv')
	folder_list.sort()
	spoof_lst, apcer, bpcer, acer, tpr = [], [], [], [], []
	for folder_idx, folder_cur in enumerate(folder_list):
		
		if args.pro == 1 and 'pro_1' not in folder_cur:
			continue
		elif args.pro == 2 and 'pro_2' not in folder_cur:
			continue
		elif args.pro == 3 and 'pro_3' not in folder_cur:
			continue
		elif args.pro not in [1,2,3]:
			raise ValueError('Invalid Protocol.')

		score_list, label_list = return_dictionary(args, [folder_cur])
		APCER, BPCER, ACER, tpr_fpr = compute_score(args, score_list, label_list, folder_cur)

		if args.pro == 2:
			spoof_type = folder_cur.split('/')[-2].split('_')[-1]
			spoof_lst.append(spoof_type)
			apcer.append(APCER)
			bpcer.append(BPCER)
			acer.append(ACER)
			tpr.append(tpr_fpr)

	if args.pro == 2:
		parse_protocol_2_result(spoof_lst, apcer, bpcer, acer, tpr)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--interval', type=float, default=0.0)
  parser.add_argument('--lr', type=float, default=3e-4,
  						help='which learning rate to measure.')
  parser.add_argument('--dataset', type=str, default='SiWM-v2', choices=['SiWM-v2', 'SiW', "OULU"],
  						help='which dataset to evaluate.')
  parser.add_argument('--pro', type=int, default=1, choices=[1,2,3,4], help='which protocol to use.')
  parser.add_argument('--weight', type=float, default=0.1, help='weight before the depth score.')
  parser.add_argument('--log_dir', type=str, default='./', 
  						help='the log directory that contains csv files.')
  args = parser.parse_args()
  main(args)