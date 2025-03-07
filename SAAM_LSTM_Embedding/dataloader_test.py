from __future__ import division
from torch.utils.data import Dataset,DataLoader
from datetime import datetime, date
from scipy import stats
import time
import math
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import logging
import os
logger = logging.getLogger('SAAM.Data')
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import random


class ClosingPrice(Dataset):
    def __init__(self, path, train_ins_num,pred_days, overlap, win_len):

        np.random.seed(0)
        self.covariate_num = 4
        self.train_ins_num = train_ins_num
        self.win_len = win_len
        print("building datasets from %s" % path)
        self.points,self.covariates,self.dates,self.withhold_len =  self.LoadData(path,pred_days,covariate_num = 4)
        print('withhold_len: ',self.withhold_len)
        #print('self.covariates', self.covariates)
        series_len = self.points.shape[0]
        self.sample_index = {}
        count = 0
        self.weight = []
        self.distance = []
        T = win_len
        self.seq_num = self.points.shape[1]
        seq_num = self.points.shape[1]
        replace = not overlap
        for i in range(seq_num):
            for head, j in enumerate(self.points[:,i]):
                if(j>0):
                    break
            indices = range(head+1,series_len+1-(T+ self.withhold_len))
            for index in indices:
                self.sample_index[count] = (i,index)
                v = np.sum(self.points[index:(index+self.win_len),i])/self.win_len+1
                self.weight.append(v)
                self.distance.append(index)
                count += 1
        if(count < train_ins_num):
            replace = True
        prob = np.array(self.weight)/sum(self.weight)
        self.dic_keys = np.random.choice(range(count),train_ins_num,replace= replace,p=prob)

        print("Maxmum traning instances",count, "Total train instances are: ", train_ins_num, ' Overlap: ',overlap, ' Replace: ',replace)

    def __len__(self):
        return self.train_ins_num

    def __getitem__(self, idx):

        T = self.win_len
        points_size = self.covariate_num + 1
        if(type(idx) != int):
            idx = idx.item()
        key = self.dic_keys[idx]
        series_id, series_index = self.sample_index[key]
        train_seq = np.zeros((self.win_len,points_size))
        try:
            train_seq[:,0] = self.points[(series_index-1):(series_index+self.win_len-1),series_id]
        except (BaseException):
            import pdb;
            pdb.set_trace()
        train_seq[:,0] = self.points[(series_index-1):(series_index+self.win_len-1),series_id]
        train_seq[:,1:] = self.covariates[series_id,series_index:(self.win_len+series_index),:]
        scaling_factor = self.weight[key]
        gt = self.points[series_index:(series_index+T),series_id]/scaling_factor

        train_seq[:,0] = train_seq[:,0]/scaling_factor
        #return (train_seq,gt,series_id, scaling_factor)
        return (train_seq, series_id, gt)

    def CalCovariate(self,input_time):
        year = input_time.year
        month = input_time.month
        day = input_time.day
        return np.array([year,month,day])


    def LoadData(self,path,pred_days,covariate_num =4):

        data = pd.read_csv(path, sep=",", index_col=0, parse_dates=True, decimal='.')
        points = data.values
        seq_num = points.shape[1]
        data_len = points.shape[0]
        pred_len =  pred_days
        withhold_len = pred_len
        dates= data.index
        covariates = np.zeros((seq_num,data_len,covariate_num))

        for idx, date in enumerate(dates):
            covariates[:,idx,:(covariate_num-1)] = self.CalCovariate(date)

        for i in range(seq_num):
            for head, j in enumerate(points[:,i]): # For each time series. we get its first non-zero value' index.
                if(j>=0):
                    break
            covariates[i,head:,covariate_num-1] = range(data_len-head)   #  Get its age feature

            for index in range(covariate_num):
                result =  stats.zscore(covariates[i,head:,index]) # We standardize all covariates to have zero mean and unit variance.
                if(np.isnan(result).any()):
                    print(points[:,i])
                    import pdb;
                    pdb.set_trace()

                else:
                    covariates[i,head:,index] = result
        return (points,covariates,dates,withhold_len)


class ClosingPriceTest(Dataset):
    def __init__(self, points,covariates, withhold_len, enc_len, dec_len):

        self.enc_len = enc_len
        self.dec_len = dec_len
        self.covariate_num = 4
        self.points, self.covariates = (points,covariates)
        seq_num = self.points.shape[1]
        rolling_times = withhold_len//dec_len
        self.test_ins_num = seq_num*rolling_times
        self.sample_index = {}
        series_len = self.points.shape[0]
        count = 0
        for i in range(seq_num):
            for j in range(rolling_times,0,-1):
                index = series_len - enc_len - j*dec_len
                self.sample_index[count] = (i,index)  # Rolling windows metrics
                count += 1
        self.count = count
        print("Data loading finished, total test instances are: ", count)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):

        T = self.enc_len + self.dec_len
        points_size = self.covariate_num + 1
        series_id, series_index = self.sample_index[idx]
        train_seq = np.zeros((T,points_size))
        #train_seq[:(self.enc_len+1),0] = self.points[series_index-1:(series_index +self.enc_len),series_id]
        #train_seq[:,1:] = self.covariates[series_id,series_index:(series_index + T),:]

        train_seq[:,0] = self.points[series_index-1:(series_index + T - 1),series_id]
        train_seq[:,1:] = self.covariates[series_id,series_index:(series_index + T),:]

        #gt = self.points[series_index+self.enc_len:(series_index+T),series_id]
        gt = self.points[series_index:(series_index + T), series_id]

        scaling_factor = np.sum(self.points[series_index:(series_index + self.enc_len),series_id])/self.enc_len+1
        train_seq[:,0] = train_seq[:,0]/scaling_factor

        return (train_seq, series_id, scaling_factor, gt)