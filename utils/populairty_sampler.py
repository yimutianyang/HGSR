import pdb
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.sparse as sp
from collections import defaultdict
import pandas as pd
import multiprocessing as mp
from time import time
import numba as nb
from numba import prange


@nb.njit()
def pop_sampling(training_user, training_item, traindata, i1, i2, i3, i4, i5, num_user, num_item):
    t = []
    for i in prange(len(training_user)):
        uid, pos_i = training_user[i], training_item[i]
        if pos_i in i1:
            j = np.random.choice(i1, 1)[0]
            while j in traindata[uid]:
                j = np.random.choice(i1, 1)[0]
            t.append([uid, pos_i+num_user, j+num_user])

        elif pos_i in i2:
            j = np.random.choice(i2, 1)[0]
            while j in traindata[uid]:
                j = np.random.choice(i2, 1)[0]
            t.append([uid, pos_i+num_user, j+num_user])


        elif pos_i in i3:
            j = np.random.choice(i3, 1)[0]
            while j in traindata[uid]:
                j = np.random.choice(i3, 1)[0]
            t.append([uid, pos_i+num_user, j+num_user])

        elif pos_i in i4:
            j = np.random.choice(i4, 1)[0]
            while j in traindata[uid]:
                j = np.random.choice(i4, 1)[0]
            t.append([uid, pos_i+num_user, j+num_user])


        elif pos_i in i5:
            j = np.random.choice(i5, 1)[0]
            while j in traindata[uid]:
                j = np.random.choice(i5, 1)[0]
            t.append([uid, pos_i+num_user, j+num_user])
    return t


@nb.njit()
def general_sampling(training_user, training_item, traindata, num_user, num_item):
    t = []
    for i in prange(len(training_user)):
        uid, pos_i = training_user[i], training_item[i]
        j = random.randint(0, num_item - 1)
        while j in traindata[uid]:
            j = random.randint(0, num_item - 1)
        t.append([uid, pos_i + num_user, j + num_user])
    return t


class Popularity_Sampler(Dataset):
    def __init__(self, traindata, num_user, num_item, neg_sample, batch_size):
        self.traindata = traindata
        self.num_user = num_user
        self.num_item = num_item
        self.neg_sample = neg_sample
        self.batch_size = batch_size
        self.compute_popularity()
        self.nbdict = nb.typed.Dict.empty(
            key_type = nb.types.int64,
            value_type = nb.types.int64[:], )
        for key, values in self.traindata.items():
            if len(values) > 0:
                self.nbdict[key] = np.asarray(list(values))


    def compute_popularity(self):
        item_users = defaultdict(list)
        item_degree = []
        self.training_user, self.training_item = [], []
        for u, items in self.traindata.items():
            self.training_user.extend([u] * len(items))
            self.training_item.extend(items)
            for i in items:
                item_users[i].append(u)
        for i, users in item_users.items():
            item_degree.append([len(users), i])
        item_degree = np.array(item_degree)
        index_sorted = np.argsort(item_degree[:, 0]) # return item id based on popularity
        length = int(len(index_sorted)/5)
        self.i1 = index_sorted[: length]
        self.i2 = index_sorted[length: length*2]
        self.i3 = index_sorted[length*2: length*3]
        self.i4 = index_sorted[length*3: length*4]
        self.i5 = index_sorted[length*4: ]
        self.head_items = self.i5
        self.tail_items = index_sorted[: length*4]

        # pdb.set_trace()

    def sample_batch_data(self, training_task, sampling_strategy):
        #------------------------------------------------- for user preference learning ----------------------------------------------#
        if training_task == 'recommendation':
            if sampling_strategy == 'pop':
                self.triplet_data = pop_sampling(nb.typed.List(self.training_user), nb.typed.List(self.training_item),
                                             self.nbdict, self.i1, self.i2, self.i3, self.i4, self.i5,
                                             self.num_user, self.num_item)
            else:
                self.triplet_data = general_sampling(nb.typed.List(self.training_user), nb.typed.List(self.training_item),
                                             self.nbdict, self.num_user, self.num_item)
        #-------------------------------------------------- for social pre-training ---------------------------------------------------#
        if training_task == 'social':
            if sampling_strategy == 'pop':
                self.triplet_data = pop_sampling(nb.typed.List(self.training_user), nb.typed.List(self.training_item),
                                             self.nbdict, nb.typed.List(self.i1), nb.typed.List(self.i2), nb.typed.List(self.i3),
                                             nb.typed.List(self.i4), nb.typed.List(self.i5), 0, self.num_user)
            else:
                self.triplet_data = general_sampling(nb.typed.List(self.training_user), nb.typed.List(self.training_item),
                                                 self.nbdict, 0, self.num_user) ### general pairwise sampling for social pre-training
        self.triplet_data = np.reshape(np.asarray(self.triplet_data), [-1, 3])
        batch_num = int(len(self.triplet_data)/self.batch_size) + 1
        indexs = np.arange(self.triplet_data.shape[0])
        np.random.shuffle(indexs)
        for k in range(batch_num):
            index_start = k*self.batch_size
            index_end = min((k+1)*self.batch_size, len(indexs))
            if index_end == len(indexs):
                index_start = len(indexs) - self.batch_size
            batch_data = self.triplet_data[indexs[index_start:index_end]]
            yield batch_data
