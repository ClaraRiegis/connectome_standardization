#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:20:08 2023

@author: clarariegis

Description: Dependence of network organisation on overall FC
            -> classification: predict pcp's diagnosis (TC or ASD) based on edge weights.
            
Plan: - 0 - FV - SLURM set-up
      - 1 - Libraries  
      - 2 - Retrieve data
      - 3 - Choose sample
          -> need to select the sample (balanced FD or full sample)
      - 4 - Stratification & parameters
      - 5 - Features: edge weights
      - 5.1 - FV - Param combinations
      - 5.2 - Training
      - 5.3 - Select the best params
      - 5.4 - Testing
          
    
    
"""

# %%                - 0 - FV - SLURM set-up

# input variable
import sys
var_iter = int(sys.argv[1])-1 # input - an index [0,12] indicating one of the pipeline combinations
#var_iter = 0 # this can be used for testing the script, for the first combination

# add current folder to path
import os
import sys
sys.path.append(os.getcwd()) 

# %%                - 1 - Libraries 


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import _pickle as cPickle
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.svm import SVC
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import bct as bct
import numpy as np
from imblearn.over_sampling import RandomOverSampler

from pathlib import Path
home_dir = str(Path.home()) # home directory
#data_dir = home_dir + '/Desktop/data_and_code'
data_dir = home_dir + '/python/classify'

import os
# os.chdir('/Users/clarariegis/Desktop/Kings/Y3_2022_23/project/publication/data and code')
os.chdir(data_dir)
from graph_measures import graph_measures

# save directory
#save_dir = home_dir+'/python/clara'
save_dir = home_dir+'/python/classify/classify_out'

# %%                - 2 - Retrieve data


all_pcp_nr = pd.read_pickle("ts_nr.pickle")
all_pcp_gsr = pd.read_pickle("ts_gsr.pickle")
balanced_dat = pd.read_pickle("clean_balanced_dat.pickle")
full_dat = pd.read_pickle("clean_unbalanced_dat.pickle")
      
# Phenotypic data:
phenotypic = pd.read_csv('Phenotypic_V1_0b_preprocessed1.csv')


# %%                - 3 - Choose sample

# X = balanced_dat
# sample = 'balanced'

# OR: 
    
X = full_dat
sample = 'full'


# %%                - 4 - Stratification & parameters




X["stratify"] = X["DX_GROUP"].astype(str) + X["SITE_ID"].astype(str)




# Divide train and test data:

df_trainval, df_test = train_test_split(X,  # Data being split.
                                        test_size=0.2,   # 80-20% separation.
                                        random_state=1,
                                        # Column use for statification:
                                        # (diagnosis-scanning site combination)
                                        stratify = X[["stratify"]])
    

# Divide the train data into 5 train/validation folds:
X = df_trainval             # Data to split.
y = df_trainval["stratify"] # Stratification target (diagnosis-site combination).
nfolds = 5                  # Number of cross-validation folds.
ncross = 5
skf = StratifiedKFold(n_splits=nfolds)
skf.get_n_splits(X, y)      # Check that the right number of folds was set.
train = []
validation = []
# seeds = np.random.randint(low = 0, high = 100, size = 5)
seeds = [55, 44, 84, 36, 70]

for i in range(ncross): 
    
    sing_train = []       # Will store the training data.
    sing_validation = []  # Will store the validation data.
    
    StratifiedKFold(n_splits=nfolds, random_state = seeds[i], shuffle=True)
    
    for train_index, validation_index in skf.split(X, y):
    
        sing_train += [X.iloc[train_index]]   # train. groups for one cross-val.
        sing_validation += [X.iloc[validation_index]] # vaildation one cross-val.
        
    train += [sing_train]
    validation += [sing_validation]




# # Checking that there is the same number of patients and sites in the groups:
# print(len(np.where(train[2]["DX_GROUP"] == 1)[0]))
# print(len(np.where(train[4]["DX_GROUP"] == 1)[0]))
# print(len(np.where(train[0]["stratify"] == "0UCLA_2")[0]))
# print(len(np.where(train[4]["stratify"] == "0UCLA_2")[0]))


# Parameters classification: 
scaler = StandardScaler()
kernels = ["rbf", "poly", "linear", "sigmoid"]
thresholds = [0.05, 0.15, 0.25]
groups = ["group_nr", "group_gsr", "group_mreg"]
c_params = np.logspace(-4, 4, 9)
gam_params = np.logspace(-4, 4, 9)



# %%                - 5 - Features: edge weights
# %%%               - 5.1 - FV - Param combinations

# FV - itetools to create parameter combinations
# from: https://stackoverflow.com/questions/71488625/how-to-create-all-possible-combinations-of-parameters-in-a-dictionaryp/71488714#71488714

import itertools

parallel_params_edge = {
    'kernels': kernels,
    'groups': groups
}

a = parallel_params_edge.values()
parallel_comb_edge = list(itertools.product(*a))

# for c in combinations:
#     LogisticRegression(penalty=c[0], class_weight=c[1], max_iter=c[2])

# %%%               - 5.2 - Training


results_list = []

columns2 = ["accuracy" , "recall" , "precision" , "average_precision", "roc_auc"]

list_plots = "vectorised matrix"



phenotypic = pd.read_csv('Phenotypic_V1_0b_preprocessed1.csv')

# Initialise a nested default dictionary to store the accuracy of the different
# function, thresholds, and folds.
accuracy = defaultdict(dict)
specificity = defaultdict(dict)
sensitivity = defaultdict(dict)
recall = defaultdict(dict)
precision = defaultdict(dict)
average_precision = defaultdict(dict)
roc_auc = defaultdict(dict)
# over_seeds = np.random.randint(0, 100, (5,5))
over_seeds = [[20, 53, 72, 20, 70],
              [44, 25,  7, 62, 15],
              [ 1, 40, 89, 81, 24],
              [30, 30, 75, 62, 34],
              [41, 26, 68, 15, 31]]


for h in range(ncross):
    
    # Initialise nested default dictionaries to store the performances.
    accuracy = defaultdict(dict)
    recall = defaultdict(dict)
    precision = defaultdict(dict)
    average_precision = defaultdict(dict)
    roc_auc = defaultdict(dict)
    
    columns = [roc_auc, average_precision, accuracy , recall , precision]

    for i in range(nfolds):  # Looping through the 5 folds.
            
        print(f"___________________ FOLD {i} ___________________")
        
        
        # –––––––––––––––––––– OVERSAMPLING ––––––––––––––––––––
        
        # Training group.
        # Get the difference of ASD vs TC participants = number of participants to add.
        diff_train = sum(train[h][i]["DX_GROUP"] == 0) - sum(train[h][i]["DX_GROUP"] == 1)
        
        
        # Random overesampling of the training group.
        oversample = RandomOverSampler(sampling_strategy='minority', random_state = over_seeds[h][i]) # oversampling strategy.
        xtrain_resampled, ytrain_resampled = oversample.fit_resample(train[h][i], train[h][i]["DX_GROUP"]) # fit and apply the transformation. xtrain_resample is the new version of the original dataframe.
        xtrain_resampled = xtrain_resampled.reset_index()
        train[h][i] = xtrain_resampled.copy()
        
        
        # –––––––––––––––––––– DATA PREPARATION ––––––––––––––––––––
        
        # Get the time series of the participants in the first fold:
        df_train_nr = all_pcp_nr[all_pcp_nr["ID"].isin(train[h][i]["subject"])]
        # Use the participants' ID as index:
        df_train_nr = df_train_nr.set_index('ID')
        # Reorder the time series so that they match the order of the training data:
        df_train_nr = df_train_nr.reindex(index=train[h][i]['subject'])
        # Resetting the index so that it's back to being consecutive numbers
        # (0, 1, ..., N):
        df_train_nr = df_train_nr.reset_index()
        # "reindex" part switch the ID columns to being "SUB_ID". It needs to be
        # switched back to "ID" for the "graph_measures" function to work:
        df_train_nr = df_train_nr.rename(columns={'subject': "ID"})
        
        
        # Doing the exact same thing, but for the GSR time series:
        df_train_gsr = all_pcp_gsr[all_pcp_gsr["ID"].isin(train[h][i]["subject"])]
        df_train_gsr = df_train_gsr.set_index('ID')
        df_train_gsr = df_train_gsr.reindex(index=train[h][i]['subject'])
        df_train_gsr = df_train_gsr.reset_index()
        df_train_gsr = df_train_gsr.rename(columns={'subject': "ID"})
        
        # Once again doing the same thing, but for the NR validation fold...
        df_val_nr = all_pcp_nr[all_pcp_nr["ID"].isin(validation[h][i]["subject"])]
        df_val_nr = df_val_nr.set_index('ID')
        df_val_nr = df_val_nr.reindex(index=validation[h][i]['subject'])
        df_val_nr = df_val_nr.reset_index()
        df_val_nr = df_val_nr.rename(columns={'subject': "ID"})
    
        # ...and for the GSR validation fold.
        df_val_gsr = all_pcp_gsr[all_pcp_gsr["ID"].isin(validation[h][i]["subject"])]
        df_val_gsr = df_val_gsr.set_index('ID')
        df_val_gsr = df_val_gsr.reindex(index=validation[h][i]['subject'])
        df_val_gsr = df_val_gsr.reset_index()
        df_val_gsr = df_val_gsr.rename(columns={'subject': "ID"})
    
    
    
       # –––––––––––––––––––– EDGE WEIGHTS ––––––––––––––––––––
    
        train_folds_nr_gsr_mreg = graph_measures(df_train_nr, df_train_gsr, prop_threshold = 0.05, phenotypic = phenotypic, exclusion = False)
       
       
       
       
        val_folds = graph_measures(df_train_nr, df_val_gsr,
                                  prop_threshold=0.05,
                                  phenotypic=phenotypic,
                                  exclusion=False,
                                  group_pred=df_val_nr)
    
        
       
       # –––––––––––––––––––– CLASSIFICATION ––––––––––––––––––––
       
        # Looping throught the pipelines:
        #for this_group in groups:
        this_group = parallel_comb_edge[var_iter][1]
        
        # Training features:
        x1 = train_folds_nr_gsr_mreg[this_group][list_plots] 
        x = pd.DataFrame()
        x["vect_matrix"] = x1.copy()
        x = pd.DataFrame(x.vect_matrix.tolist(), index= x.index)
       
        this_kernel = parallel_comb_edge[var_iter][0]
    
        for this_c in c_params: 
            
            for this_gam in gam_params:
                
                if this_kernel == 'poly':
                    svc = SVC(kernel = this_kernel, C = this_c, gamma = this_gam, probability=True, max_iter=1000)  
                else:
                    svc = SVC(kernel = this_kernel, C = this_c, gamma = this_gam, probability=True)
                
                # Combining the scaler and the rbf SVM into a pipeline:
                pipe_svm = make_pipeline(scaler, svc)
               
                pipe_svm.fit(x, train[h][i]["DX_GROUP"])
                
                if this_group == "group_mreg" : 
                    
                    # Validation features:
                    val_folds1 = val_folds["group_pred"][list_plots].copy()
                    val_folds2 = pd.DataFrame()
                    val_folds2["vect_matrix"] = val_folds1.copy()
                    val_folds2 = pd.DataFrame(val_folds2.vect_matrix.tolist(), index= val_folds2.index)
                    
                    
                    # Prediction using the validation fold:
                    val_pred = pipe_svm.predict(val_folds2) 
                    # Proba 
                    val_prob = pipe_svm.predict_proba(val_folds2) 
                    
                else:
                    
                    val_folds1 = val_folds[this_group][list_plots].copy()
                    val_folds2 = pd.DataFrame()
                    val_folds2["vect_matrix"] = val_folds1.copy()
                    val_folds2 = pd.DataFrame(val_folds2.vect_matrix.tolist(), index= val_folds2.index)
                    
                    # Prediction using the validation fold:
                    val_pred = pipe_svm.predict(val_folds2) 
                    # Proba 
                    val_prob = pipe_svm.predict_proba(val_folds2)
                    
    
                # –––––––––––––––––––– PERFORMANCE ––––––––––––––––––––
                
                combination = (this_kernel + "_" + "C" + str(this_c) + "_" + "Gam" + str(this_gam) + "_" + this_group)
    
                # Storing the accuracy score for every pipeline:
                accuracy[combination][i] = accuracy_score(validation[h][i]["DX_GROUP"], val_pred)
                # Recall score:
                recall[combination][i] = recall_score(validation[h][i]["DX_GROUP"], val_pred)
                # Precision score:
                precision[combination][i] = precision_score(validation[h][i]["DX_GROUP"], val_pred)
                # Average precision score:
                average_precision[combination][i] = average_precision_score(validation[h][i]["DX_GROUP"], val_prob[:,1])
                # Roc Auc curve:
                roc_auc[combination][i] = roc_auc_score(validation[h][i]["DX_GROUP"], val_prob[:,1]) 
                
                print("___________________ ONE PARAMETER ___________________")
    
    
    df_all_results = pd.DataFrame()
    for i in range(len(columns)):                 # Loop through the 7 scores. 
        scores_df = pd.DataFrame(columns[i])      # Convert defaultdict into df. 
        scores_df = scores_df.transpose()         # (folds, pipeline) -> (pipeline, fold).
        scores_df = np.mean(scores_df, axis = 1)  # Average the 5 folds.
        df_all_results[columns2[i]] = scores_df   # Add the averages in a dataframe.


    # Saving the results:
    #df_all_results.to_pickle("/Users/clarariegis/Desktop/Kings/Y3_2022_23/project/classif_results/train_edges_" + sample + "_fold_avr.pkl")
    #df_all_results.to_pickle(save_dir+"/edge/train_edges_" + str(var_iter) + "_fold_avr.pkl")
    df_all_results.to_pickle(save_dir+"/train_edges_" + sample + "_iter_" + str(var_iter) + "_crossval_" + str(h) + "_fold_avr.pkl")
    
    results_list.append(df_all_results)
            



# %%%               - 5.3 - Select the best params

# Average the results of the 5 cross-validations:
all_cross_results = pd.concat(results_list).groupby(level=0).mean()

# Order the results by ROC AUC score:
all_cross_results = all_cross_results.sort_values('roc_auc', ascending=False)


# Select row with the highest roc_auc -> select the C and gamma best parameters.
max_grid = all_cross_results.roc_auc.idxmax().split('_')


# Parameters to use for the testing:
    # - kernel, threshold and standardization remain the same as in the training.
    # - Need to extract the C and gamma that led to the highest ROC AUC.
test_gamma = pd.to_numeric(max_grid[2][3:]) # Extract the gamma value and getting rid of "gam" (e.g. Gam0.45 -> 0.45).
test_c = pd.to_numeric(max_grid[1][1:])     # Same thing for the C parameter (e.g. C1.1 -> 1.1).


# %%%               - 5.4 - Testing



# Get the time series of the participants in the first fold:
df_train_nr = all_pcp_nr[all_pcp_nr["ID"].isin(df_trainval["subject"])]
# Use the participants' ID as index:
df_train_nr = df_train_nr.set_index('ID')
# Reorder the time series so that they match the order of the training data:
df_train_nr = df_train_nr.reindex(index=df_trainval['subject'])
# Resetting the index so that it's back to being consecutive numbers
# (0, 1, ..., N):
df_train_nr = df_train_nr.reset_index()
# "reindex" part switch the ID columns to being "SUB_ID". It needs to be
# switched back to "ID" for the "graph_measures" function to work:
df_train_nr = df_train_nr.rename(columns={'subject': "ID"})

# Doing the exact same thing, but for the GSR time series:
df_train_gsr = all_pcp_gsr[all_pcp_gsr["ID"].isin(df_trainval["subject"])]
df_train_gsr = df_train_gsr.set_index('ID')
df_train_gsr = df_train_gsr.reindex(index=df_trainval['subject'])
df_train_gsr = df_train_gsr.reset_index()
df_train_gsr = df_train_gsr.rename(columns={'subject': "ID"})

# Once again doing the same thing, but for the NR validation fold...
df_val_nr = all_pcp_nr[all_pcp_nr["ID"].isin(df_test["subject"])]
df_val_nr = df_val_nr.set_index('ID')
df_val_nr = df_val_nr.reindex(index=df_test['subject'])
df_val_nr = df_val_nr.reset_index()
df_val_nr = df_val_nr.rename(columns={'subject': "ID"})

# ...and for the GSR validation fold.
df_val_gsr = all_pcp_gsr[all_pcp_gsr["ID"].isin(df_test["subject"])]
df_val_gsr = df_val_gsr.set_index('ID')
df_val_gsr = df_val_gsr.reindex(index=df_test['subject'])
df_val_gsr = df_val_gsr.reset_index()
df_val_gsr = df_val_gsr.rename(columns={'subject': "ID"})



# –––––––––––––––––––– EDGE WEIGHTS ––––––––––––––––––––

trainval_folds_nr_gsr_mreg = {}
test_folds = {}


    
    
# Training measures:
trainval_folds_nr_gsr_mreg = graph_measures(df_train_nr, df_train_gsr, 
                                            prop_threshold = 0.05, 
                                            phenotypic = phenotypic, 
                                            exclusion = False)

    
# Test measures:
test_folds = graph_measures(df_train_nr, df_val_gsr, 
                            prop_threshold = 0.05, 
                            phenotypic = phenotypic,exclusion=False, 
                            group_pred = df_val_nr)




    
    
x = trainval_folds_nr_gsr_mreg[this_group][list_plots] 

# Turn the lists into columns:
# Training features:
x1 = trainval_folds_nr_gsr_mreg[this_group][list_plots].copy()
x = pd.DataFrame()
x["vect_matrix"] = x1.copy()
x = pd.DataFrame(x.vect_matrix.tolist(), index= x.index)


# set maximum number of iterations for polynomial kernel (to stop script running indefinitely)
if this_kernel == 'poly':
    svc = SVC(kernel = this_kernel, C = test_c, gamma = test_gamma, probability=True, max_iter=1000) 
else:
    svc = SVC(kernel = this_kernel, C = test_c, gamma = test_gamma, probability=True) 

# Combining the scaler and the rbf SVM into a pipeline:
pipe_svm = make_pipeline(scaler, svc)


pipe_svm.fit(x, df_trainval["DX_GROUP"])


if this_group == "group_mreg" : 
    
    # Validation features:
    test_folds1 = test_folds["group_pred"][list_plots].copy()
    test_folds = pd.DataFrame()
    test_folds["vect_matrix"] = test_folds1.copy()
    test_folds = pd.DataFrame(test_folds.vect_matrix.tolist(), index= test_folds.index)
    
    # Prediction using the validation fold:
    val_pred = pipe_svm.predict(test_folds)
    # Proba 
    val_prob = pipe_svm.predict_proba(test_folds)
    
else:
    
    # Validation features:
    test_folds1 = test_folds[this_group][list_plots].copy()
    test_folds = pd.DataFrame()
    test_folds["vect_matrix"] = test_folds1.copy()
    test_folds = pd.DataFrame(test_folds.vect_matrix.tolist(), index= test_folds.index)
    
    # Prediction using the validation fold:
    val_pred = pipe_svm.predict(test_folds)
    # Proba 
    val_prob = pipe_svm.predict_proba(test_folds)
    


# –––––––––––––––––––– PERFORMANCE ––––––––––––––––––––

# Storing the accuracy score for every pipeline:
accuracy_test = accuracy_score(df_test["DX_GROUP"], val_pred)
# Recall score:
recall_test = recall_score(df_test["DX_GROUP"], val_pred)
# Precision score:
precision_test = precision_score(df_test["DX_GROUP"], val_pred)
# Average precision score:
average_precision_test = average_precision_score(df_test["DX_GROUP"], val_prob[:,1])
# Roc Auc curve:
roc_auc_test = roc_auc_score(df_test["DX_GROUP"], val_prob[:,1])


# Results:
test_results = {"kernel": this_kernel, 
                   "c" : test_c, 
                   "gamma" : test_gamma, 
                   "group" : this_group,
                   "roc_auc": roc_auc_test, 
                   "average_precision": average_precision_test, 
                   "accuracy" : accuracy_test , 
                   "recall" : recall_test , 
                   "precision" : precision_test}

# convert dict to dataframe
df_test_results = pd.DataFrame.from_dict(test_results, orient='index', columns=['iter_'+str(var_iter)])

#df_all_results.to_pickle("/Users/clarariegis/Desktop/Kings/Y3_2022_23/project/classif_results/test_edges_" + sample + ".pkl")
df_test_results.to_pickle(save_dir+"/test_edges_" + sample + "_iter_" + str(var_iter) + ".pkl")

# %% test conversion of results dict to pandas dataframe

# # Results:
# this_kernel = 'rbf'
# test_c = 1
# test_gamma = 1
# this_group = 'group_nr'
# roc_auc_test = 1
# average_precision_test = 1
# accuracy_test = 1
# recall_test = 1
# precision_test = 1

# test_results = {"kernel": this_kernel, 
#                    "c" : test_c, 
#                    "gamma" : test_gamma, 
#                    "group" : this_group,
#                    "roc_auc": roc_auc_test, 
#                    "average_precision": average_precision_test, 
#                    "accuracy" : accuracy_test , 
#                    "recall" : recall_test , 
#                    "precision" : precision_test}

# df_test_results = pd.DataFrame.from_dict(test_results, orient='index', columns=['iter_'+str(var_iter)])

# print(df_test_results.to_string())
