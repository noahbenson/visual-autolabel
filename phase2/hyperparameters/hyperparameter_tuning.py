#!/usr/bin/env python
# coding: utf-8

# In[1]:


# IMPORTS
import os, sys, json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import torch
import optuna

import neuropythy as ny

import visual_autolabel as va
import visual_autolabel.benson2025 as project


# In[2]:


# PATH CONFIGURATION
data_root = '/gscratch/nbenlab/data/visual-autolabel'
dataset_cache_path = f'{data_root}/datasets'
model_cache_path = f'{data_root}/models/'
figures_path = f'{data_root}/figures'


# In[3]:


# MORE PATH CONFIGURATION
project.config.dataset_cache_path = dataset_cache_path
project.config.model_cache_path = model_cache_path
project.config.figures_path = figures_path


# In[5]:


# PARTITION WE WANT TO USE
trn = np.array([102311, 102816, 104416, 105923, 108323, 109123, 111312, 111514,
        114823, 115017, 115825, 118225, 126426, 128935, 130114, 130518,
        132118, 140117, 145834, 146432, 146735, 146937, 150423, 156334,
        157336, 158035, 158136, 159239, 162935, 164131, 164636, 165436,
        167036, 167440, 169040, 169444, 169747, 171633, 172130, 175237,
        176542, 177140, 177746, 178142, 178243, 181232, 181636, 182739,
        185442, 186949, 187345, 191336, 191841, 192641, 193845, 195041,
        197348, 198653, 199655, 200210, 200311, 200614, 201515, 203418,
        204521, 205220, 214524, 221319, 233326, 239136, 246133, 249947,
        263436, 283543, 320826, 330324, 352738, 360030, 365343, 380036,
        385046, 393247, 397760, 401422, 406836, 412528, 429040, 436845,
        463040, 467351, 525541, 550439, 552241, 562345, 572045, 573249,
        581450, 601127, 617748, 638049, 671855, 680957, 690152, 724446,
        725751, 732243, 751550, 757764, 765864, 771354, 782561, 783462,
        789373, 818859, 826353, 859671, 871762, 872764, 878776, 878877,
        899885, 901139, 901442, 905147, 910241, 942658, 958976, 966975,
        973770, 995174])
val = np.array([814649, 381038, 116726, 100610, 770352, 971160, 541943, 134829,
        585256, 131217, 209228, 926862, 146129, 135124, 927359, 212419,
        825048, 257845, 318637, 898176, 389357, 654552, 536647, 173334,
        180533, 155938, 547046, 191033, 346137, 395756, 196144, 137128,
        627549])
# partition = {'trn':trn, 'val':val}
partition = (trn,val)


# In[6]:


# HYPERPARAMETER TUNING SET UP
def objective(trial, inputs, partition):
#     model = trial.suggest_categorical("model", ['resnet18', 'resnet34'])
    lr = trial.suggest_float("lr", 0.001, 0.01)
    gamma = trial.suggest_float("gamma", 0.8, 1)
    bce_weight = trial.suggest_float("bce_weight", 0.5, 1)
    batch_size = trial.suggest_int("batch_size", 1, 60)
    frac_lr_1 = trial.suggest_float("frac_lr_1", 0.1, 1)
    frac_bce_1 = trial.suggest_float("frac_bce_1", 0.1, 1)
    frac_lr_2 = trial.suggest_float("frac_lr_2", 0.1, 1)
    frac_bce_2 = trial.suggest_float("frac_bce_2", 0, 1)
    
    training_plan = [
        {'lr': lr, 'gamma': gamma, 'bce_weight': bce_weight}, 
        {'lr': lr * frac_lr_1, 'gamma': gamma, 'bce_weight': bce_weight * frac_bce_1}, 
        {'lr': lr * frac_lr_1 * frac_lr_2, 'gamma': gamma, 'bce_weight': bce_weight * frac_bce_1 * frac_bce_2}
    ]
    
    training_history = va.train.train_until(
        in_features=inputs,
        out_features='area',
        dataloaders= project.hcp.make_dataloaders(in_features=inputs, out_features='area',partition=partition),
        model_cache_path=model_cache_path,
        dataset_cache_path=dataset_cache_path,
        device='cuda',
        model_key=f'hV4_VO1_VO2_optuna_{inputs}_{trial.number}',
        model=va.image.UNet,
        training_plan=training_plan,
        partition=partition,
        until=2
    )
    
    print(training_history)
    
    training_history_df = pd.DataFrame(training_history)
    loss = training_history_df['dice'].min()
    
    return loss


# In[ ]:


# LAUNCH OPTIMIZATION
# As it stands, optuna cannot accept an objective() function that has more than one argument.
# So, we are going to implement this post's trick to work around it: https://www.kaggle.com/discussions/general/261870
def run_optuna(inputs, partition):
    modified_objective = lambda trial: objective(trial, inputs=inputs, partition=partition)
    n_trials = 100
    
    study = optuna.create_study(direction='minimize')
    study.optimize(modified_objective, n_trials=100)
    
    trial_results = study.trials_dataframe()
#     trial_results = trial_results.loc[trial_results['state'] == 'COMPLETE']
#     trial_results = trial_results.drop(['state'], axis=1)
    trial_results.to_csv(f'{data_root}/models/hV4_VO1_VO2_optuna_{inputs}_results.csv', index=False)
    
    best_trial = study.best_trial
    
    return best_trial


# In[ ]:


# CALLING THE FUNCTION run_optuna()
inputs_list = ['anat', 'func']

for inputs in inputs_list:
    run_optuna(inputs, partition)


# In[7]:


# LAUNCH OPTIMIZATION
# As it stands, optuna cannot accept an objective() function that has more than one argument.
# So, we are going to implement this post's trick to work around it: https://www.kaggle.com/discussions/general/261870
def run_optuna(inputs, partition):
    modified_objective = lambda trial: objective(trial, inputs=inputs, partition=partition)
    n_trials = 75
    
    study = optuna.create_study(direction='minimize')
    study.optimize(modified_objective, n_trials=75)
    
    trial_results = study.trials_dataframe()
#     trial_results = trial_results.loc[trial_results['state'] == 'COMPLETE']
#     trial_results = trial_results.drop(['state'], axis=1)
    trial_results.to_csv(f'{data_root}/models/hV4_VO1_VO2_optuna_{inputs}_results.csv', index=False)
    
    best_trial = study.best_trial
    
    return best_trial


# In[ ]:


run_optuna('func', partition)


# In[4]:


# CHECKS

print("islink:", os.path.islink("/gscratch/nbenlab/data/visual-autolabel/models"))
print("lexists:", os.path.lexists("/gscratch/nbenlab/data/visual-autolabel/models"))
print("exists:", os.path.exists("/gscratch/nbenlab/data/visual-autolabel/models"))

if os.path.islink("/gscratch/nbenlab/data/visual-autolabel/models"):
    print("target:", os.readlink("/gscratch/nbenlab/data/visual-autolabel/models"))
