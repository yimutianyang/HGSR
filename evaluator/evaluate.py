# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 23:12:38 2019

@author: Administrator
"""

import math 
import numpy as np
from sklearn.metrics import average_precision_score
import pdb
from collections import defaultdict
import multiprocessing as mp
import numba
from numba import njit


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def get_idcg(length):
    idcg = 0.0
    for i in range(length):
        idcg = idcg + math.log(2) / math.log(i + 2)
    return idcg

########################################### test part ############################################
def _init(_test_ratings, _train_ratings, _topk_list, _predictions, _itemset):
    global test_ratings, train_ratings, topk_list, predictions, itemset
    test_ratings = _test_ratings
    train_ratings = _train_ratings
    topk_list = _topk_list
    predictions = _predictions
    itemset = _itemset


def get_one_performance(_uid):
    u = _uid
    metrics = {}
    pos_index = list(test_ratings[u])
    pos_length = len(test_ratings[u])
    if pos_length > 0:
        neg_index = list(itemset-set(train_ratings[u])-set(test_ratings[u]))
        pos_index.extend(neg_index)
        pre_one = predictions[u][pos_index]
        indices = largest_indices(pre_one, topk_list[-1])
        indices = list(indices[0])
        for topk in topk_list:
            hit_value = 0
            dcg_value = 0
            for idx in range(topk):
                ranking = indices[idx]
                if ranking < pos_length:
                    hit_value += 1
                    dcg_value += math.log(2) / math.log(idx + 2)
            target_length = min(topk, pos_length)
            hr_cur = hit_value / target_length
            recall_cur = hit_value / pos_length
            ndcg_cur = dcg_value / get_idcg(target_length)
            metrics[topk] = {'hr': hr_cur, 'recall': recall_cur, 'ndcg': ndcg_cur}
    return metrics


def evaluate(_testdata, _traindata, _topk_list, _predictions, _testuser):
    _item_count = _predictions.shape[1]
    _itemset = set(range(_item_count))
    hr_topk_list = defaultdict(list)
    recall_topk_list = defaultdict(list)
    ndcg_topk_list = defaultdict(list)
    hr_out, recall_out, ndcg_out = {}, {}, {}
    # test_users = _testdata.keys()
    test_users = _testuser
    with mp.Pool(processes=10, initializer=_init, initargs=(_testdata, _traindata, _topk_list, _predictions, _itemset)) as pool:
        all_metrics = pool.map(get_one_performance, test_users)
    for i, one_metrics in enumerate(all_metrics):
        for topk in _topk_list:
            hr_topk_list[topk].append(one_metrics[topk]['hr'])
            recall_topk_list[topk].append(one_metrics[topk]['recall'])
            ndcg_topk_list[topk].append(one_metrics[topk]['ndcg'])
    for topk in _topk_list:
        hr_out[topk] = np.mean(hr_topk_list[topk])
        recall_out[topk] = np.mean(recall_topk_list[topk])
        ndcg_out[topk] = np.mean(ndcg_topk_list[topk])
    return recall_out, ndcg_out



def _init_tail(_test_ratings, _train_ratings, _topk_list, _predictions, _itemset, _head_item, _tail_item):
    global test_ratings, train_ratings, topk_list, predictions, itemset, head_item, tail_item
    test_ratings = _test_ratings
    train_ratings = _train_ratings
    topk_list = _topk_list
    predictions = _predictions
    itemset = _itemset
    head_item = _head_item
    tail_item = _tail_item


def get_one_performance_head_tail(_uid):
    u = _uid
    metrics = {}
    pos_index = list(test_ratings[u])
    pos_length = len(test_ratings[u])
    if pos_length > 0:
        neg_index = list(itemset-set(train_ratings[u])-set(test_ratings[u]))
        pos_index.extend(neg_index)
        pre_one = predictions[u][pos_index]
        indices = largest_indices(pre_one, topk_list[-1])
        indices = list(indices[0])
        for topk in topk_list:
            hit_head, hit_tail = 0, 0
            dcg_head, dcg_tail = 0, 0
            for idx in range(topk):
                ranking = indices[idx]
                if ranking < pos_length:
                    if pos_index[ranking] in head_item:
                        hit_head += 1
                        dcg_head += math.log(2) / math.log(idx + 2)
                    if pos_index[ranking] in tail_item:
                        hit_tail += 1
                        dcg_tail += math.log(2) / math.log(idx + 2)
            target_length = min(topk, pos_length)
            hr_head = hit_head / target_length
            recall_head = hit_head / pos_length
            ndcg_head = dcg_head / get_idcg(target_length)
            hr_tail = hit_tail / target_length
            recall_tail = hit_tail / pos_length
            ndcg_tail = dcg_tail / get_idcg(target_length)
            metrics[topk] = {'hr_head': hr_head, 'hr_tail': hr_tail, 'recall_head': recall_head, 'recall_tail': recall_tail,
                            'ndcg_head': ndcg_head, 'ndcg_tail': ndcg_tail}
    return metrics


def evaluate_head_tail(_testdata, _traindata, _topk_list, _predictions,
                       head_item, tail_item, cur_users):
    _itemset = set(range(_predictions.shape[1]))
    hr_topk_head, hr_topk_tail = defaultdict(list), defaultdict(list)
    recall_topk_head, recall_topk_tail = defaultdict(list), defaultdict(list)
    ndcg_topk_head, ndcg_topk_tail = defaultdict(list), defaultdict(list)
    hr_head, recall_head, ndcg_head = {}, {}, {}
    hr_tail, recall_tail, ndcg_tail = {} ,{} ,{}
    test_users = []
    for u in cur_users:
        if u in _testdata.keys():
            test_users.append(u)

    ###   evaluate head users   ###
    with mp.Pool(processes=10, initializer=_init_tail,
                 initargs=(_testdata, _traindata, _topk_list, _predictions, _itemset, head_item, tail_item)) as pool:
        all_metrics = pool.map(get_one_performance_head_tail, test_users)
    for i, one_metrics in enumerate(all_metrics):
        for topk in _topk_list:
            hr_topk_head[topk].append(one_metrics[topk]['hr_head'])
            recall_topk_head[topk].append(one_metrics[topk]['recall_head'])
            ndcg_topk_head[topk].append(one_metrics[topk]['ndcg_head'])
            hr_topk_tail[topk].append(one_metrics[topk]['hr_tail'])
            recall_topk_tail[topk].append(one_metrics[topk]['recall_tail'])
            ndcg_topk_tail[topk].append(one_metrics[topk]['ndcg_tail'])
    for topk in _topk_list:
        hr_head[topk] = np.mean(hr_topk_head[topk])
        recall_head[topk] = np.mean(recall_topk_head[topk])
        ndcg_head[topk] = np.mean(ndcg_topk_head[topk])
        hr_tail[topk] = np.mean(hr_topk_tail[topk])
        recall_tail[topk] = np.mean(recall_topk_tail[topk])
        ndcg_tail[topk] = np.mean(ndcg_topk_tail[topk])

    '''
    ###   evaluate tail users   ###
    hr_topk_head_u2, hr_topk_tail_u2 = defaultdict(list), defaultdict(list)
    recall_topk_head_u2, recall_topk_tail_u2 = defaultdict(list), defaultdict(list)
    ndcg_topk_head_u2, ndcg_topk_tail_u2 = defaultdict(list), defaultdict(list)
    hr_head_u2, recall_head_u2, ndcg_head_u2 = {}, {}, {}
    hr_tail_u2, recall_tail_u2, ndcg_tail_u2 = {} ,{} ,{}
    with mp.Pool(processes=10, initializer=_init,
                 initargs=(_testdata, _traindata, _topk_list, _predictions, _itemset, head_item, tail_item)) as pool:
        all_metrics_u2 = pool.map(get_one_performance_head_tail, filter_tail_user)

    for i, one_metrics in enumerate(all_metrics_u2):
        for topk in _topk_list:
            hr_topk_head_u2[topk].append(one_metrics[topk]['hr_head'])
            recall_topk_head_u2[topk].append(one_metrics[topk]['recall_head'])
            ndcg_topk_head_u2[topk].append(one_metrics[topk]['ndcg_head'])
            hr_topk_tail_u2[topk].append(one_metrics[topk]['hr_tail'])
            recall_topk_tail_u2[topk].append(one_metrics[topk]['recall_tail'])
            ndcg_topk_tail_u2[topk].append(one_metrics[topk]['ndcg_tail'])
    for topk in _topk_list:
        hr_head_u2[topk] = np.mean(hr_topk_head_u2[topk])
        recall_head_u2[topk] = np.mean(recall_topk_head_u2[topk])
        ndcg_head_u2[topk] = np.mean(ndcg_topk_head_u2[topk])
        hr_tail_u2[topk] = np.mean(hr_topk_tail_u2[topk])
        recall_tail_u2[topk] = np.mean(recall_topk_tail_u2[topk])
        ndcg_tail_u2[topk] = np.mean(ndcg_topk_tail_u2[topk])
    '''
    return recall_head, ndcg_head, recall_tail, ndcg_tail #recall_head_u2, ndcg_head_u2, recall_tail_u2, ndcg_tail_u2


def get_map(pred, label):
    '''
    计算MAP
    '''
    ap_list = []
    for i in range(label.shape[0]):
        if 1 in label[i]:
            y_true = label[i]
            y_predict = pred[i]
            precision = average_precision_score(y_true, y_predict)
            ap_list.append(precision)
    mean_ap = sum(ap_list)/len(ap_list)
    return round(mean_ap,4)

def get_precise(pre_matrix, gt_matrix):
    '''
    计算ACC
    '''
    pre_precise = []
    for i in range(pre_matrix.shape[0]):
        if 1 in gt_matrix[i]:              
            index = np.where(gt_matrix[i]==1)[0][0]
            if pre_matrix[i][index] == max(pre_matrix[i]):
                pre_precise.append(1)
            else:
                pre_precise.append(0)
    mean_pre_precise = sum(pre_precise)/len(pre_precise)
    return round(mean_pre_precise,4)




#existing_label1_list = np.load('../data/existing_label1_list.npy').tolist()
#existing_label2_list = np.load('../data/existing_label2_list.npy').tolist()
#existing_label3_list = np.load('../data/existing_label3_list.npy').tolist()
#itemset = set(range(33899))
#missing_label1_list = list(itemset.symmetric_difference(existing_label1_list))
#missing_label2_list = list(itemset.symmetric_difference(existing_label2_list))
#missing_label3_list = list(itemset.symmetric_difference(existing_label3_list))

#gt = np.load('./data/all_attributes.npy')
#layer4 = np.load('./save_npy/test_layer4_item_attributes_infer.npy')
#lp_pred = np.load('./lp_data/133_loss9.605365e-09_item_atributes.npy')
#pred = np.load('epoch_66_item_attributes_infer.npy')
#pred_1 = np.load('epoch_64_item_attributes_infer.npy')
#pred_2 = np.load('epoch_74_item_attributes_infer.npy')
#pred_3 = np.load('epoch_77_item_attributes_infer.npy')
#pred_4 = np.load('epoch_86_item_attributes_infer.npy')
#pred_5 = np.load('epoch_91_item_attributes_infer.npy')
#pred_6 = np.load('epoch_96_item_attributes_infer.npy')
#lp_pred_20 = np.load('./lp_data/85_loss8.9048376e-07_item_atributes.npy')  #K=20 label propagation 效果最好
def compute_mmm(x, y):
    a = get_map(x[missing_label1_list,:14], y[missing_label1_list, :14])
    b = get_map(x[missing_label2_list, 14:66], y[missing_label2_list, 14:66])
    c = get_precise(x[missing_label3_list, 66:], y[missing_label3_list, 66:])
    print(a, b, c)


#compute_mmm(lp_pred_20, gt)
#compute_mmm(pred, gt)
#compute_mmm(layer4, gt) #best
#compute_mmm(lp_pred, gt)
#compute_mmm(pred_2, gt)
#compute_mmm(pred_3, gt)
#compute_mmm(pred_4, gt)
#compute_mmm(pred_5, gt)
#compute_mmm(pred_6, gt)


