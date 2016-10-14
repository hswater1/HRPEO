# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 18:37:54 2016

@author: shen
"""

import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
import numpy as np

##load the dat
df_train = pd.read_csv('training_dataset.csv.zip', compression="zip")
df_test = pd.read_csv('test_dataset.csv.zip', compression="zip")
df_train.info()

hk_num = df_train['user_id'].value_counts()
hk_open = df_train.groupby(['user_id'])['opened'].agg('sum')
hk_clk = df_train.groupby(['user_id'])['clicked'].agg('sum')

hk_rates = pd.concat([hk_num,hk_open,hk_clk],axis=1)
hk_rates.rename(columns={'user_id':'num_emails','opened':'num_open','clicked':'num_clk'},inplace=True)
hk_rates['open_rate']=hk_rates['num_open']/hk_rates['num_emails']
hk_rates['clk_rate']=hk_rates['num_clk']/hk_rates['num_emails']

##get the training data size
tr_size=df_train.shape[0]

y = df_train['opened'].astype(int)

##prepare the features
df_all = pd.concat([df_train,df_test], axis=0).drop(['opened','open_time',
                       'clicked','click_time','unsubscribed','unsubscribe_time',
                       'mail_id'], axis=1)

##fill the missing values
df_all['hacker_timezone'].fillna(99999, inplace=True)
df_all['last_online'].fillna(df_all.hacker_created_at, inplace=True)
df_all['mail_category'].fillna('other_category', inplace=True)
df_all['mail_type'].fillna('other_type', inplace=True)

df_all['hacker_timezone']=df_all['hacker_timezone'].astype(int)
df_all['last_online']=df_all['last_online'].astype(int)

##
hk_num = df_train['user_id'].value_counts()
hk_open = df_train.groupby(['user_id'])['opened'].agg('sum')
hk_clk = df_train.groupby(['user_id'])['clicked'].agg('sum')

hk_rates = pd.concat([hk_num,hk_open,hk_clk],axis=1)
hk_rates.rename(columns={'user_id':'num_emails','opened':'num_open','clicked':'num_clk'},inplace=True)
hk_rates['open_rate']=hk_rates['num_open']/hk_rates['num_emails']
hk_rates['clk_rate']=hk_rates['num_clk']/hk_rates['num_emails']


df_all=pd.merge(df_all,hk_rates,left_on='user_id', right_index=True,
                how='left', sort=False)
                
##perform one hot encoding
one_hot=pd.get_dummies(df_all['mail_category'])
df_all=pd.concat([df_all,one_hot],axis=1)

one_hot=pd.get_dummies(df_all['mail_type'])
df_all=pd.concat([df_all,one_hot],axis=1)

one_hot=pd.get_dummies(df_all['hacker_confirmation'])
df_all=pd.concat([df_all,one_hot],axis=1)

one_hot=pd.get_dummies(df_all['hacker_timezone'])
df_all=df_all.drop(['mail_category','mail_type','hacker_confirmation','hacker_timezone'],axis=1)
df_all=pd.concat([df_all,one_hot],axis=1)

df_all=df_all.drop(['user_id'],axis=1)

X=df_all[:tr_size]
test=df_all[tr_size:]

##creat the validation data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1729)

##adapt the metric function for xgboost 
def f_score(y_pred,y_true):
    print(y_pred)
    return 'f1', f1_score(y_true.get_label(),y_pred)

##apply xgboost 
dm_train = xgb.DMatrix(X_train, label=y_train)
dm_test = xgb.DMatrix(test)
dm_valid = xgb.DMatrix(X_test, label=y_test)
train = xgb.DMatrix(X, label=y)

evals_result = {}
clf = xgb.train(
            {'max_depth': 12, 'min_child_weight': 1, 'eta': 0.03, 'objective': 'multi:softmax',
             'num_class': 2, 'booster': 'gbtree'},
            train, num_boost_round=1000, feval = f_score, maximize=True, 
            evals=[(dm_valid,'eval'), (dm_train,'train')], evals_result=evals_result,
            early_stopping_rounds=30)

y_pred = clf.predict(dm_test)

##create the submission csv file 
np.savetxt("output02.csv",y_pred.astype(int), fmt='%i') 

