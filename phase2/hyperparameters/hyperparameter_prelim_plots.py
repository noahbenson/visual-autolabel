#!/usr/bin/env python
# coding: utf-8

# In[1]:


# IMPORTS
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data_root = '/gscratch/nbenlab/data/visual-autolabel/models'


# In[3]:


# INITIALIZING DICTS

anat_trials = []
func_trials = []
dice_anat_1 = []
dice_anat_2 = []
dice_func_1 = []
dice_func_2 = []


# In[4]:


# ITERATING THROUGH SUB-DIRS

for folder in os.listdir(data_root):
    folder_path = os.path.join(data_root, folder)
    
    if os.path.isdir(folder_path):
        try:
            trial_num = folder[24:]
            tsv_path = os.path.join(folder_path, "training.tsv")
            
            df = pd.read_csv(tsv_path, sep='\t')
            dice_loss_col = df.iloc[:, 1].values
            
            if "anat" in str(folder_path):
                dice_anat_1.append(dice_loss_col[0])
                dice_anat_2.append(dice_loss_col[1])
                anat_trials.append(trial_num)
            else:
                dice_func_1.append(dice_loss_col[0])
                dice_func_2.append(dice_loss_col[1])
                func_trials.append(trial_num)
            
        except Exception as e:
            print(f"Skipping folder {folder}: {e}")


# In[5]:


# SORT THE DATA BY TRIAL NUMBERS
anat_sorted_data = sorted(zip(anat_trials, dice_anat_1, dice_anat_2))
func_sorted_data = sorted(zip(func_trials, dice_func_1, dice_func_2))


# In[13]:


# UNZIP AND PLOT ANATOMICAL DATA
anat_trials, dice_anat_1, dice_anat_2 = zip(*anat_sorted_data)

plt.figure(figsize=(20,5))
plt.plot(anat_trials, dice_anat_1, marker='o', color='purple', label= "Dice Loss 1")
plt.plot(anat_trials, dice_anat_2, marker='o', color='yellow', label= "Dice Loss 2")

plt.xlabel("Trial Number")
plt.ylabel("Dice Loss")
plt.title("Dice Loss vs Trial Number")

plt.show()


# In[11]:


# UNZIP AND PLOT ANATOMICAL DATA
func_trials, dice_func_1, dice_func_2 = zip(*func_sorted_data)

plt.figure(figsize=(20,5))
plt.plot(func_trials, dice_func_1, marker='o', color='purple', label= "Dice Loss 1")
plt.plot(func_trials, dice_func_2, marker='o', color='yellow', label= "Dice Loss 2")

plt.xlabel("Trial Number")
plt.ylabel("Dice Loss")
plt.title("Dice Loss vs Trial Number")

plt.show()

