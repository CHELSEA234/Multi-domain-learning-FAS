import csv
import numpy as np
from glob import glob
from sklearn.metrics import roc_curve, auc

def compute_tpr_fpr_(given_pos_score, given_neg_score, rate=0.002):
	given_pos_score.sort()
	given_neg_score.sort()
	idx = int(rate * len(given_neg_score))
	# print(idx)
	new_threshold = given_neg_score[-(idx+1)]
	# print(given_neg_score[-5:])
	# print(new_threshold)
	# import sys;sys.exit(0)
	correct_counter = 0
	for _ in given_pos_score:
		if _ > new_threshold:
			correct_counter += 1
	res = correct_counter/len(given_pos_score)
	return res

def ACER_compute(idx_eer):	
	_tpr, _fpr = tpr[idx_eer], fpr[idx_eer]
	_tnr, _fnr = tnr[idx_eer], fnr[idx_eer]
	assert _tpr + _fnr == 1, print(_tpr, _fnr)
	assert _tnr + _fpr == 1, print(_tnr, _fpr)

	APCER = _fpr/(_fpr+_tnr)
	BPCER = _fnr/(_fnr+_tpr)
	ACER  = 0.5 * (APCER+BPCER)
	return ACER, APCER, BPCER

def compute_result(pos_score_list, neg_score_list, csv_file_name):
	y_true = [1]*len(pos_score_list) + [0]*len(neg_score_list)
	y_probas = pos_score_list + neg_score_list

	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	fpr, tpr, scores = roc_curve(y_true, y_probas)

	# EER result.
	fnr = 1 - tpr
	tnr = 1 - fpr
	EER0 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
	EER1 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
	EER = min(EER0, EER1)

	min_error = len(scores)
	min_ACC = 0.0 
	min_ACER = 0.0
	min_APCER = 0.0
	min_BPCER = 0.0
	
	for cur_threshold in scores:
		false_neg_count, false_pos_count = 0, 0
		for idx, score_cur in enumerate(y_probas):
			label_cur = y_true[idx]
			# print(label_cur, type(label_cur), score_cur)
			# import sys;sys.exit(0)
			if score_cur <= cur_threshold and label_cur == 1:
				false_neg_count += 1
			if score_cur > cur_threshold and label_cur == 0:
				false_pos_count += 1
		APCER = false_pos_count / len(neg_score_list)
		BPCER = false_neg_count / len(pos_score_list)
		ACER = 0.5 * (APCER + BPCER)
		if min_error > ACER:
			min_error = ACER
			min_ACER = ACER
			min_APCER = APCER
			min_BPCER = BPCER

	## why does your output are all the same??
	## https://github.com/ZitongYu/CDCN/blob/master/CVPR2020_paper_codes/utils.py#L94
	# print(f"threshold is {cur_threshold:.3f}")
	# print(f"the EER is {EER:.3f}.")
	# print(f"the ACER result is {min_ACER:.3f}, the BPCER is {min_BPCER:.3f}, the APCER is {min_APCER:.3f}.")
	# print()
	return EER*100, min_ACER*100, min_APCER*100, min_BPCER*100

def compute_tpr_fpr_(given_pos_score, given_neg_score, rate=0.002):
	given_pos_score.sort()
	given_neg_score.sort()
	idx = int(rate * len(given_neg_score))
	# print(idx)
	new_threshold = given_neg_score[-(idx+1)]
	# print(given_neg_score[-5:])
	# print(new_threshold)
	# import sys;sys.exit(0)
	correct_counter = 0
	for _ in given_pos_score:
		if _ > new_threshold:
			correct_counter += 1
	res = correct_counter/len(given_pos_score)
	return res

def get_trunc_auc(y_true, y_probas, fpr_value):
	fpr, tpr, _ = roc_curve(y_true, y_probas)
	roc_auc = auc(fpr, tpr)

	idx = fpr <= fpr_value
	area_curve = sum(tpr[idx])
	tot_area = sum(np.ones_like(tpr)[idx])
	if tot_area == 0:
		raise ZeroDivisionError('when computing truncated ROC aread')
	t_auc = area_curve/tot_area
	return t_auc

csv_file_name_list = [
					"_data_all_stage_ft_type_spoof_decay_1_epoch_60_lr_1e-07_spoof_region_architecture_epoch_num_1.csv",
					"_data_all_stage_ft_type_age_decay_1_epoch_60_lr_1e-07_spoof_region_architecture_epoch_num_59.csv",
					"_data_all_stage_ft_type_race_decay_1_epoch_60_lr_1e-07_spoof_region_architecture_epoch_num_1.csv",
					"_data_all_stage_ft_type_illu_decay_1_epoch_60_lr_1e-07_spoof_region_architecture_epoch_num_59.csv"
					]
TARGET_DATASET_LIST = ["SIW", "Oulu", 'SIWM']
sample_counter = 0
last_sample_name = ""
for csv_file_name in csv_file_name_list:
	csv_file = open(csv_file_name, 'r')
	csv_reader = csv.reader(csv_file, delimiter=',')
	line_count = 0
	test_A_label_list = []
	test_A_pos_score_list, test_A_neg_score_list = [], []
	Oulu_pos_score_list = []
	test_B_label_list = []
	test_B_pos_score_list, test_B_neg_score_list = [], []
	Oulu_neg_score_list = []
	ill_flag = 'illu' in csv_file_name
	for idx, row in enumerate(csv_reader):
		if line_count == 0:
			print(row)
			pass
		else:
			dataset_name = row[1]
			depth_score = float(row[-4])
			p_score = float(row[-5])
			region_map = float(row[-3])
			label_cur = float(row[-2])
			dataset_ = row[-1]
			# print(row)
			# import sys;sys.exit(0)
			if dataset_name in TARGET_DATASET_LIST:
				if "test_A" in dataset_:
					score_cur = depth_score + 5*p_score
					if dataset_name != 'SIWM':
						sample_name = row[0].split('/')[-2]
						if last_sample_name != sample_name:
							sample_counter = 0
							last_sample_name = sample_name
						else:
							sample_counter += 1
						if sample_counter < 1 and dataset_name == 'Oulu':
							pass
						elif sample_counter < 1 and dataset_name == 'SIW':
							pass
						else:
							continue
					if label_cur == 1:
						test_A_pos_score_list.append(score_cur)
					elif label_cur == 0:
						test_A_neg_score_list.append(score_cur)
				elif "test_B" in dataset_:
					score_cur = depth_score + 15*p_score
					if dataset_name != 'SIWM':
						sample_name = row[0].split('/')[-2]
						if last_sample_name != sample_name:
							sample_counter = 0
							last_sample_name = sample_name
						else:
							sample_counter += 1
					if label_cur == 1:
						if dataset_name == "Oulu" and ill_flag == True:
							Oulu_pos_score_list.append(score_cur)
						else:
							test_B_pos_score_list.append(score_cur)
					elif label_cur == 0:
						if dataset_name == "Oulu" and ill_flag == True:
							Oulu_neg_score_list.append(score_cur)
						else:
							test_B_neg_score_list.append(score_cur)
		line_count += 1
	csv_file.close()
	# break
	# print(TARGET_DATASET_LIST)
	print("A", len(test_A_pos_score_list)+len(test_A_neg_score_list))
	print("B", len(test_B_pos_score_list)+len(test_B_neg_score_list))
	# print(len(Oulu_neg_score_list[:250])+len(Oulu_pos_score_list[:250]))
	if ill_flag == True:
		test_B_pos_score_list = test_B_pos_score_list+Oulu_pos_score_list[:200]
		test_B_neg_score_list = test_B_neg_score_list+Oulu_neg_score_list[:150]
		print("B", len(test_B_pos_score_list)+len(test_B_neg_score_list))

	# import sys;sys.exit(0)
	# rate_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
	rate_list = [0.01, 0.001]
	print(f"{csv_file_name}")
	print(f"The target dataset is {TARGET_DATASET_LIST}")
	for rate_cur in rate_list:
		res_ = compute_tpr_fpr_(test_A_pos_score_list, test_A_neg_score_list,rate=rate_cur)
		print(f"in the testA, the tpr_fpr_{rate_cur*100}% is {res_:.3f}.")
		res_ = compute_tpr_fpr_(test_B_pos_score_list, test_B_neg_score_list,rate=rate_cur)
		print(f"in the testB, the tpr_fpr_{rate_cur*100}% is {res_:.3f}.")
		print("============================================================")

	EER_A, min_ACER_A, min_APCER_A, min_BPCER_A = compute_result(test_A_pos_score_list, test_A_neg_score_list, csv_file_name)
	# print("Test B statistic: ")
	EER_B, min_ACER_B, min_APCER_B, min_BPCER_B = compute_result(test_B_pos_score_list, test_B_neg_score_list, csv_file_name)
	# print(f"threshold is {cur_threshold:.3f}")
	print(f"the EER is {EER_A:.1f}/{EER_B:.1f}.")
	print(f"the ACER is {min_ACER_A:.1f}/{min_ACER_B:.1f}.")
	print(f"the APCER is {min_APCER_A:.1f}/{min_APCER_B:.1f}.")
	print(f"the BPCER is {min_BPCER_A:.1f}/{min_BPCER_B:.1f}.")
	print()

	# # for fpr_value in [0.1, 0.01]:
	# # 	y_true = [1] * len(test_A_pos_score_list) + [0] * len(test_A_neg_score_list) 
	# # 	y_probas = test_A_pos_score_list + test_A_neg_score_list
	# # 	t_auc = get_trunc_auc(y_true, y_probas, fpr_value)
	# # 	print(f"in the testA, the truncated_AUC_{fpr_value*100}% is {t_auc:.3f}.")
	# # 	y_true = [1] * len(test_B_pos_score_list) + [0] * len(test_B_neg_score_list) 
	# # 	y_probas = test_B_pos_score_list + test_B_neg_score_list
	# # 	t_auc = get_trunc_auc(y_true, y_probas, fpr_value)
	# # 	print(f"in the testB, the truncated_AUC_{fpr_value*100}% is {t_auc:.3f}.")
