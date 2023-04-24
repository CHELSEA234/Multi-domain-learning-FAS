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
from sklearn import metrics
import numpy as np

def get_tpr_at_fpr(tpr_lst, fpr_lst, score_lst, fpr_value):
	"""returns true postive rate and threshold given false positive rate value."""
	abs_fpr = np.absolute(fpr_lst - fpr_value)
	idx_min = np.argmin(abs_fpr)
	fpr_value_target = fpr_lst[idx_min]
	idx = np.max(np.where(fpr_lst == fpr_value_target))
	return tpr_lst[idx], score_lst[idx]

def my_metrics(label_list, pred_list, val_phase=False):
	"""
	computes FAS metrics. 
		Parameters: 
			val_phase (bool): flag for train and test stage.
	"""
	fpr, tpr, scores = metrics.roc_curve(label_list,pred_list,
										 drop_intermediate=True)
	auc_score = metrics.auc(fpr,tpr)
	fnr = 1 - tpr
	tnr = 1 - fpr
	EER0 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
	EER1 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
	EER  = min(EER0, EER1) 
	best_ACER, best_AP, best_BP = 100, 100, 100
	best_threshold = 100
	for idx_ in range(len(tpr)):		
		_tpr, _fpr = tpr[idx_], fpr[idx_]
		_tnr, _fnr = tnr[idx_], fnr[idx_]
		assert _tpr + _fnr == 1, print(_tpr, _fnr)
		assert _tnr + _fpr == 1, print(_tnr, _fpr)
		# https://chalearnlap.cvc.uab.cat/challenge/33/track/33/metrics/
		APCER = _fpr/(_fpr+_tnr)
		BPCER = _fnr/(_fnr+_tpr)
		ACER  = 0.5 * (APCER+BPCER)
		if ACER < best_ACER:
			best_ACER = ACER
			best_AP   = APCER
			best_BP   = BPCER
			best_threshold = scores[idx_]

	## fnr == 0.5% as the first PAMI paper version. 
	abs_fnr = np.absolute(fnr - 0.005)
	idx = np.argmin(abs_fnr)
	res_tpr = tpr[idx]
	if not val_phase:
		tpr_h, _ = get_tpr_at_fpr(tpr, fpr, scores, 0.005)
		tpr_m, _ = get_tpr_at_fpr(tpr, fpr, scores, 0.01)	
		tpr_l, _ = get_tpr_at_fpr(tpr, fpr, scores, 0.02)	
		return best_AP, best_BP, best_ACER, EER, res_tpr, auc_score, [tpr_h, tpr_m, tpr_l]
	else:
		return best_AP, best_BP, best_ACER, EER, res_tpr, auc_score