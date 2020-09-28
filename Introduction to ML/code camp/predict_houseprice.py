#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:30:40 2020

@author: tsheringlhamo
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mserr
from sklearn.decomposition import PCA

trainx_df = pd.read_csv("train.csv",index_col = 'Id')
testx_df = pd.read_csv("test.csv", index_col = 'Id')
trainy_df = trainx_df['SalePrice'] #y column
trainx_df.drop('SalePrice', axis=1, inplace = True) # axis 1 means colwise, inplace true create new df
#print(train_df.shape)
#print(train_df.isnull().sum())

sample_size = len(trainx_df) #length of training data
#print(sample_size)
#ratio of cols with null values
col_with_nulls = [[col, float(trainx_df[col].isnull().sum())/float(sample_size)] 
                  for col in trainx_df.columns if trainx_df[col].isnull().sum()]
#print(col_with_nulls)
col_to_drop = [x for (x,y) in col_with_nulls if y > .3] # select cols to drop with ratio >.3
#print(col_to_drop)
trainx_df.drop(col_to_drop, axis=1, inplace= True)
testx_df.drop(col_to_drop, axis=1, inplace = True)
#print(len(trainx_df.columns))
#remove sample with missing values
trainx_df.dropna(axis = 0, inplace= True) #remove rows

#for all na in categorical features, create new feature as na

categorical_columns =[col for col in trainx_df.columns if trainx_df[col].dtype == object]
#print(len(categorical_columns))
#categorical_columns.append('MSSubClass') #add mssubclass which is ordical type 
#print(len(categorical_columns))
ordinal_columns = [col for col in trainx_df.columns if col not in categorical_columns]
#print(len(ordinal_columns))

dummy_row = list()
for col in trainx_df.columns:  
    if col in categorical_columns: #for each categorical col create one dummy row
        dummy_row.append("dummy")
    else:
        dummy_row.append("")

new_row = pd.DataFrame([dummy_row], columns = trainx_df.columns)
trainx_df = pd.concat([trainx_df, new_row], axis = 0, ignore_index = True)
testx_df = pd.concat([testx_df], axis = 0, ignore_index = True)

for col in categorical_columns: #replace na with dummy
    trainx_df[col].fillna(value = 'dummy', inplace = True)
    testx_df[col].fillna(value = 'dummy', inplace = True)
    
enc = OneHotEncoder(drop='first', sparse=False)
enc.fit(trainx_df[categorical_columns])
#print(enc.get_feature_names(categorical_columns))
trainx_enc = pd.DataFrame(enc.transform(trainx_df[categorical_columns]))
print(trainx_df.shape)
#testx_enc = pd.DataFrame(enc.transform(testx_df[categorical_columns]))
#print(trainx_df.shape)
trainx_enc.columns = enc.get_feature_names(categorical_columns)
#testx_enc.columns = enc.get_feature_names(categorical_columns)
trains_df = pd.concat([trainx_df[ordinal_columns], trainx_enc], axis = 1, ignore_index = True)
#testx_df = pd.concat([testx_df[ordinal_columns], testx_enc], axis = 1, ignore_index = True)

trainx_df.drop(trainx_df.tail(1).index, inplace= True)

imputer = KNNImputer(n_neighbors=2)
imputer.fit(trainx_df)
trainx_df_filled = imputer.transform(trainx_df)
trainx_df_filled = pd.DataFrame(trainx_df_filled, columns = trainx_df.columns)
trainx_df_filled.reset_index(drop = True, inplace = True)

print(trainx_df_filled.isnull().sum())

testx_df_filled = imputer.transform(testx_df)
testx_df_filled = pd.DataFrame(testx_df_filled, columns = testx_df.columns)
testx_df_filled.reset_index(drop = True, inplace = True)












