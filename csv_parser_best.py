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
	return res, new_threshold

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

csv_file_name_list = ["_data_all_stage_ft_type_illu_decay_1_epoch_60_lr_1e-07_spoof_region_architecture_epoch_num_59.csv"]
TARGET_DATASET_LIST = ["SIW", "Oulu", 'SIWM']
score_mode = 'new_mode' # 'fine_tune_mode' 

sample_name_lst = []
sample_name_lst_2 = []

dmap_pred_list_pos, dmap_pred_list_neg = [], []
p_list_pos, p_list_neg = [], []
region_map_list_pos, region_map_list_neg = [], []
# A = np.linspace(0.0, 10.0, num=100)
# score_list = np.linspace(1,100,num=100)
score_list = np.linspace(0,100,num=500)
score_list = np.linspace(0,100,num=500)
max_score = 0
best_coeff = 0
# rate_list = [0.01, 0.005, 0.002]
rate_list = [0.001]

for csv_file_name in csv_file_name_list:
	csv_file = open(csv_file_name, 'r')
	csv_reader = csv.reader(csv_file, delimiter=',')
	line_count = 0
	test_A_label_list = []
	test_A_pos_score_list, test_A_neg_score_list = [], []
	test_B_label_list = []
	test_B_pos_score_list, test_B_neg_score_list = [], []
	for idx, row in enumerate(csv_reader):
		if line_count == 0:
			# print(row)
			pass
		else:
			# print(row)
			image_name = row[0].split('/')[-1]
			p_pretrain = float(row[2])
			dmap_pre_pretrain = float(row[3])
			region_map_pretrain = float(row[4])
			p = float(row[5])
			dmap_pred = float(row[6])
			region_map = float(row[7])
			# if score_mode == 'new_mode':
			# 	score_cur = dmap_pred + coeff*region_map
			label_cur = float(row[-2])
			dataset_name = row[1]
			dataset_ = row[-1]
			# print(row)
			# import sys;sys.exit(0)
			if dataset_name in TARGET_DATASET_LIST:
				# if image_name not in sample_name_lst:
				# 	sample_name_lst.append(image_name)
				# else:
				# 	continue
				if "test_B" == dataset_:
					if label_cur == 1:
						dmap_pred_list_pos.append(dmap_pred)
						p_list_pos.append(p)
						region_map_list_pos.append(region_map)
						# print(dmap_pred + 10*p, dmap_pred, p)
						# import sys;sys.exit(0)
					elif label_cur == 0:
						dmap_pred_list_neg.append(dmap_pred)
						p_list_neg.append(p)
						region_map_list_neg.append(region_map)
		line_count += 1
	csv_file.close()

print("B dataset number: ", len(p_list_neg)+len(p_list_pos))
import sys;sys.exit(0)
# coeff: 132.33233233233233, 0.4173 ==> test_A.
for coeff in score_list:
	for coeff_depth in score_list:
		pos_score_list, neg_score_list = [], []




		for i in range(len(p_list_pos)):
			p_score = p_list_pos[i]
			depth_score = dmap_pred_list_pos[i]
			region_map = region_map_list_pos[i] 
			# print(p_score, region_map, p_score)
			# import sys;sys.exit(0)
			score = depth_score + 10*p_score
			# score = coeff_depth*depth_score + coeff*region_map
			pos_score_list.append(score)

		for i in range(len(p_list_neg)):
			p_score = p_list_neg[i]
			depth_score = dmap_pred_list_neg[i]
			region_map = region_map_list_neg[i] 
			score = depth_score + 10*p_score
			# score = coeff_depth*depth_score + coeff*region_map
			neg_score_list.append(score)

		assert len(p_list_pos) == len(pos_score_list)
		assert len(p_list_neg) == len(neg_score_list)

		for rate_cur in rate_list:
			res_, threshold_ = compute_tpr_fpr_(pos_score_list, neg_score_list,rate=rate_cur)
			print(f"in the test, the tpr_fpr_{rate_cur*100}% is {res_:.3f}: on threshold {threshold_:.3f} at value {coeff:.3f}.")
			if res_ > max_score:
				max_score = res_
				best_coeff = coeff
				best_coeff_depth = coeff_depth
		import sys;sys.exit(0)

print(max_score)
print(best_coeff)
print(best_coeff_depth)

