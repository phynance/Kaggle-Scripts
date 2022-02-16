import pandas as pd
import numpy as np
from datetime import datetime
from lightgbm import LGBMRegressor
import gresearch_crypto
import traceback
import time
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
#from sklearn.metrics import make_scorer
#import seaborn as sns
import MovingAverageClass

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



### Loading datasets
path = "/kaggle/input/g-research-crypto-forecasting/"
df_train = pd.read_csv(path + "train.csv")
df_test = pd.read_csv(path + "example_test.csv")
df_asset_details = pd.read_csv(path + "asset_details.csv")
df_supp_train = pd.read_csv(path + "supplemental_train.csv")


df_asset_details.sort_values(["Asset_ID"])

#supplemental_train.csv is the same as above (train.csv) except contains less rows.

traindtype={'timestamp': 'int32', 'Asset_ID': 'int8', 'Count': 'int16', 'Open': 'float16', 'High': 'float16', 'Low': 'float16', 'Close': 'float16',
       'Volume': 'float32', 'VWAP': 'float16','Target': 'float16'}



df_trainFull = pd.concat([df_train,df_supp_train])   ## to train the model completely, combine with moving average feature
#df_train = df_train.astype(traindtype)
df_train = df_trainFull.astype(traindtype)

############################################################################################################################
# Data Cleaning
    
df_train.replace([np.inf, -np.inf], np.nan, inplace=True)

df_train['VWAP'] = np.where(df_train.VWAP.isnull(), (df_train["Close"]+df_train["Open"])/2, df_train.VWAP)

df_train = df_train.dropna(how="any", axis = 0) # remove rows



############################################################################################################################
# Feature Engineering

def COHL(df): 
    rawCOHL = (df['Close']-df['Open'])/(df['High'] - df['Low'])  
    rawCOHL.fillna(1.0, inplace = True)
    return rawCOHL
        
def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])
def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']

def GK_vol(df):
    return ((1 / 2 * np.log(df['High']/df['Low']) ** 2 - (2 * np.log(2) - 1) * np.log(df['Close']/df['Open']) ** 2)) * 10000  # *10000 because too small value

def RS_vol(df):
    return (np.log(df['High']/df['Close'])*np.log(df['High']/df['Open']) + np.log(df['Low']/df['Close'])*np.log(df['Low']/df['Open'])) * 10000

    
def get_features(df):
    ###  for getting features in training data
    df_feat = df[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']].copy()
    df_feat['GK_vol'] = GK_vol(df_feat)
    df_feat['RS_vol'] = RS_vol(df_feat)
    df_feat['log_ret'] = df_feat['Close']/df_feat['Open']
    df_feat['COHL_ratio'] = COHL(df_feat)
    
    df_feat['base'] = df_feat.Close
    df_feat['Open'] = df_feat.Open/df_feat.Close
    df_feat['High'] = df_feat.High/df_feat.Close
    df_feat['Low'] = df_feat.Low/df_feat.Close
    df_feat['VWAP'] = df_feat.VWAP/df_feat.Close
    df_feat['VolumePerTrade'] = df_feat.Volume  / df_feat.Count
    
    
    df_feat['pandas_SMA_15'] = df_feat.Close.rolling(window=15,min_periods=1).mean()   ### first 15 rows are simply aggregate mean, need to redefine a function (must first)
    df_feat['SMA_15_percentDiff'] = (df_feat.Close - df_feat.pandas_SMA_15)/df_feat.pandas_SMA_15 * 100
    
    df_feat['Close'] = 1
    df_feat['Upper_Shadow'] = upper_shadow(df_feat)
    df_feat['Lower_Shadow'] = lower_shadow(df_feat) 

    df_feat.drop(['Close'], axis=1,inplace=True)
    return df_feat

def get_features_iter(df,assetID):
    df_feat = df[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']].copy()
    df_feat['GK_vol'] = GK_vol(df_feat)
    df_feat['RS_vol'] = RS_vol(df_feat)
    df_feat['log_ret'] = df_feat['Close']/df_feat['Open']
    df_feat['COHL_ratio'] = COHL(df_feat)
    
    df_feat['base'] = df_feat.Close
    df_feat['Open'] = df_feat.Open/df_feat.Close
    df_feat['High'] = df_feat.High/df_feat.Close
    df_feat['Low'] = df_feat.Low/df_feat.Close
    df_feat['VWAP'] = df_feat.VWAP/df_feat.Close
    df_feat['VolumePerTrade'] = df_feat.Volume  / df_feat.Count
    
    
    ################################################################################
    dict_MAclass[assetID].push(df_feat.iloc[0]['Close'])
    df_feat['pandas_SMA_15'] = dict_MAclass[assetID].get_mean()   ### first 15 rows are simply aggregate mean, need to redefine a function (must first)
    ################################################################################
    
    df_feat['SMA_15_percentDiff'] = (df_feat.Close - df_feat.pandas_SMA_15)/df_feat.pandas_SMA_15 * 100
    df_feat['Upper_Shadow'] = df_feat['High'] - np.maximum(1, df_feat['Open'])
    df_feat['Lower_Shadow'] = np.minimum(1, df['Open']) - df_feat['Low'] 

    
    df_feat.drop(['Close'], axis=1,inplace=True)

    return df_feat
    
######### HyperParameter Tunned Results (done in a separate script)  #############
DEVICE = "CPU" # CPU

models = {
2: LGBMRegressor(learning_rate=0.05, max_depth=10, n_estimators=500, num_leaves=200, device = DEVICE), #0.048186375269837343
0: LGBMRegressor(learning_rate=0.05, max_depth=10, n_estimators=1000, num_leaves=300, device = DEVICE), #0.011239773361252904  
1: LGBMRegressor(learning_rate=0.05, max_depth=10, n_estimators=500, num_leaves=200, device = DEVICE),#0.019377907309737395
5: LGBMRegressor(learning_rate=0.05, max_depth=10, n_estimators=500, num_leaves=500, device = DEVICE), #0.031013536303656584
7: LGBMRegressor(learning_rate=0.05, max_depth=70, n_estimators=1000, num_leaves=400, device = DEVICE), #0.015706439422546833
6: LGBMRegressor(learning_rate=0.05, max_depth=10, n_estimators=500, num_leaves=500, device = DEVICE), #0.02345612371174138    
9: LGBMRegressor(learning_rate=0.05, max_depth=30, n_estimators=500, num_leaves=400, device = DEVICE), #0.02586388368632851    
11: LGBMRegressor(learning_rate=0.05, max_depth=70, n_estimators=500,num_leaves=100, device = DEVICE), #0.016762287930919273     
13: LGBMRegressor(learning_rate=0.05, max_depth=10, n_estimators=500,num_leaves=200, device = DEVICE), #0.026184514321744297    
12: LGBMRegressor(learning_rate=0.05, max_depth=10, n_estimators=500,num_leaves=100, device = DEVICE), #0.017003778362314828
3: LGBMRegressor(learning_rate=0.05, max_depth=10, n_estimators=500, num_leaves=400, device = DEVICE), #0.0019081766003313284
8: LGBMRegressor(learning_rate=0.05, max_depth=70, n_estimators=500, num_leaves=100, device = DEVICE), #0.015322020443587976  
10: LGBMRegressor(learning_rate=0.05, max_depth=70, n_estimators=500,num_leaves=100, device = DEVICE), #0.021674358244667458   
4: LGBMRegressor(learning_rate=0.05, max_depth=30, n_estimators=500, num_leaves=100, device = DEVICE), #0.01948345322325262
}

############################################################################################################################
def get_Xy_and_model_for_asset(df_train, asset_id):
    df = df_train[df_train["Asset_ID"] == asset_id]
    
    df_proc = get_features(df)
    df_proc = df_proc.dropna(how="any", axis = 0) # remove rows
    df_proc['y'] = df['Target']

    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]
    
    model = models[asset_id] 
      
    return X, y, model

############################## Training LGBM models ##############################

Xs = {}
ys = {}
trainedModel = {}


for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
    print(f"Trained model for {asset_name:<16} (ID={asset_id:<2})")
    X, y, model = get_Xy_and_model_for_asset(df_train, asset_id)       
    try:
        trainedModel[asset_id] = model.fit(X,y)
    except: 
        trainedModel[asset_id] =  None 


############################## save the model as a pre-trained model ##############################

import joblib

filename = "Completed_All_LGBMmodel_500trees.joblib"
joblib.dump(trainedModel, filename)
