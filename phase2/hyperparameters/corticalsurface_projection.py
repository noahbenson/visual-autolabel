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
import seaborn as sns

import torch
import optuna

import neuropythy as ny

import visual_autolabel as va
import visual_autolabel.benson2025 as project
from visual_autolabel.benson2025.hcp import HCPDataset


# In[2]:


# MANUALLY CONFIGURING NEUROPYTHY PATHS, JUST IN CASE. 
ny.config["freesurfer_subject_paths"] =  "/gscratch/nbenlab/data/freesurfer/subjects"
ny.config["data_cache_root"] = "/gscratch/nbenlab/data/neuropythy"
ny.config["hcp_subject_paths"] = "/gscratch/nbenlab/data/hcp/subjects"
ny.config["hcp_auto_download"] = True
ny.config["hcp_lines_path"] = "/gscratch/nbenlab/data/hcp/lines"
ny.config["hcp_metadata_path"] = "/gscratch/nbenlab/data/hcp/meta"
ny.config["visual_performance_fields_path"] = "/gscratch/nbenlab/data/performance-fields"


# In[3]:


# CONFIGURING VISUAL AUTOLABEL PATHS. 
data_root = '/gscratch/nbenlab/data/visual-autolabel'
dataset_cache_path = f'{data_root}/datasets/'
model_cache_path = f'{data_root}/models'
figures_path = f'{data_root}/figures'

project.config.dataset_cache_path = dataset_cache_path
project.config.model_cache_path = model_cache_path
project.config.figures_path = figures_path


# In[4]:


# NOT THE DATASET WE ARE WORKING ON. 
val = np.array(814649, 381038, 116726, 100610, 770352, 971160, 541943, 134829,
         585256, 131217, 209228, 926862, 146129, 135124, 927359, 212419,
         825048, 257845, 318637, 898176, 389357, 654552, 536647, 173334,
         180533, 155938, 547046, 191033, 346137, 395756, 196144, 137128,
         627549)


# In[ ]:


# sid = 182739
# hem = 'lh'

# labels_gold = ny.load(f'/gscratch/nbenlab/data/hcp/labels/visual/{hem}.{sid}.mgz')


# In[5]:


# WEIGHTS OF THE BEST MODEL FROM THE OPTUNA RUN
weights_file = f'{model_cache_path}/hV4_VO1_VO2_optuna_anat_11/best_inputs.pt'


# In[6]:


# LOADING BEST MODEL FROM OPTUNA RUN
weights = torch.load(weights_file, weights_only=True, map_location=torch.device('cpu'))
model = va.UNet(feature_count=7, segment_count=3)
model.load_state_dict(weights)


# In[7]:


im = HCPDataset(
    inputs='anat',
    outputs='area',
    sids=val,
    cache_path = os.path.join(dataset_cache_path, 'HCP'),
    hemis='lr'
)


# In[ ]:


# sid = 115017
# ny.config = "/gscratch/nbenlab/data/hcp/subjects"
# sub = ny.data['hcp_lines'].subjects[sid]


# In[ ]:


# MODEL PREDICTIONS ON SINGULAR SUBJECT
k=0
target = im.targets[k]
sid = target['subject']
sub = ny.data['hcp_lines'].subjects[sid]
labels = im.predlabels(k=0, model=model)
fmap = im.image_cache.get_flatmap(target=target, view={'hemisphere': 'rh'})


# In[ ]:


# MODEL PREDICTIONS ON SINGULAR SUBJECT (CONT'D)
hemi_labels = np.zeros(sub.rh.vertex_count)
hemi_labels[fmap.labels] = labels[1] # SO, HEMI-LABELS HAS THE SAME LENGTH AS THE LOADED LABELS FROM THE FILES
# lh = sub.lh.with_prop(visual_area=hemi_labels) # THIS IS USED TO PLOT THE 3D VERSION OF THE CORTEX
# ny.cortex_plot(lh, color='visual_area') # THIS IS USED TO PLOT THE 3D VERSION OF THE CORTEX


# In[ ]:


# 2D PLOT
ny.cortex_plot(fmap, color=labels[1])


# In[ ]:


# print(len(hemi_labels))


# In[ ]:





# In[ ]:


# fmap.vertex_count


# In[ ]:


# im.targets


# In[ ]:


rater = 'R4'
sid = 115017
hem = 'lh'
labels = ny.load(f"/gscratch/nbenlab/data/hcp/labels/crcns2021/{rater}/{sid}/{hem}.ventral_label.mgz")


# In[ ]:


# print(np.unique(labels))


# In[ ]:


# print(np.unique(hemi_labels))


# In[ ]:





# In[8]:


val = np.array(814649, 381038, 116726, 100610, 770352, 971160, 541943, 134829,
         585256, 131217, 209228, 926862, 146129, 135124, 927359, 212419,
         825048, 257845, 318637, 898176, 389357, 654552, 536647, 173334,
         180533, 155938, 547046, 191033, 346137, 395756, 196144, 137128,
         627549)
raters = ['R1', 'R2', 'R3', 'R4', 'R5']
areas = ['hV4', 'VO1', 'VO2']

rater_map = {
    'R1': 'A7',
    'R2': 'A8',
    'R3': 'A11',
    'R4': 'A12',
    'R5': 'A13'
}


# In[9]:


def calculate_dice_score(ground_truth, hemi_labels, area_index):
    intersection = 0
    ground_truth_labels_size = 0
    predicted_labels_size = 0

    for true_label, predicted_label in zip(ground_truth, hemi_labels):
        if true_label == (area_index + 4):
            ground_truth_labels_size += 1
        if predicted_label == (area_index + 1):
            predicted_labels_size += 1
        if (true_label == (area_index + 4)) and (predicted_label == (area_index + 1)):
            intersection += 1
            
    dice_score = (2 * intersection)/(ground_truth_labels_size + predicted_labels_size)
    
    return dice_score


# In[10]:


# THIS CELL IS THE COMPLETE ONE. CALCULATES THE DICE LOSS AND ALSO CREATES THE DATAFRAME.

rows = []

for sid in val:
#     print("SID:", sid)
    for rater in raters:
        for hem in ['lh', 'rh']:
            # TODO: Load in rater subject label
            ground_truth = ny.load(f"/gscratch/nbenlab/data/hcp/labels/crcns2021/{rater}/{sid}/{hem}.ventral_label.mgz")
            # TODO: Load in lh rh model predictions
            target, index = im._to_target_index({'rater': rater_map[rater], 'subject': sid})
            # TEST: Print k
#             print(f'k: {index}')
            sub = ny.data['hcp_lines'].subjects[sid]
            labels = im.predlabels(k=index, model=model)
            fmap = im.image_cache.get_flatmap(target=target, view={'hemisphere': hem})
            if hem == 'lh':
                hemi_labels = np.zeros(sub.lh.vertex_count)
                hemi_labels[fmap.labels] = labels[0]
            elif hem == 'rh':
                hemi_labels = np.zeros(sub.rh.vertex_count)
                hemi_labels[fmap.labels] = labels[1]
            
            # TEST: Print both lengths out
#             print(f'Ground Truth: {len(ground_truth)}')
#             print(f'Hemi-Label: {len(hemi_labels)}')
            
            for area in areas:
                # TODO: Find intersection betweeen the ground truth and hemi labels
                dice_score = calculate_dice_score(ground_truth, hemi_labels, areas.index(area))
                rows.append([rater, sid, hem, area, dice_score])
#                 print(f'Rater: {rater}, SID: {sid}, Hem: {hem}, Area: {area}, Dice Score: {dice_score}')

df = pd.DataFrame(rows, columns=["Rater", "SID", "Hemisphere", "Visual Area", "Dice Score"])
df.to_csv(os.path.join(model_cache_path, 'anat_rater_model_dice.csv'), index=False)


# In[5]:


anat_df = pd.read_csv(os.path.join(model_cache_path, 'anat_rater_model_dice.csv'))
func_df = pd.read_csv(os.path.join(model_cache_path, 'func_rater_model_dice.csv'))


# In[6]:


anat_df.head()


# In[7]:


func_df.head()


# In[8]:


anat_df.tail()


# In[9]:


func_df.tail()


# In[10]:


anat_rater_avg = (
    anat_df.groupby("Rater")["Dice Score"]
      .mean()
      .sort_values(ascending=False)
)

print(anat_rater_avg)


# In[11]:


func_rater_avg = (
    func_df.groupby("Rater")["Dice Score"]
      .mean()
      .sort_values(ascending=False)
)

print(func_rater_avg)


# In[12]:


anat_visual_avg = (
    anat_df.groupby("Visual Area")["Dice Score"]
      .mean()
      .sort_values(ascending=False)
)

print(anat_visual_avg)


# In[13]:


func_visual_avg = (
    func_df.groupby("Visual Area")["Dice Score"]
      .mean()
      .sort_values(ascending=False)
)

print(func_visual_avg)


# In[14]:


anat_std_err_df = anat_df.groupby('Visual Area')['Dice Score'].agg(
    mean='mean',
    std_error=lambda x: np.std(x, ddof=1) / np.sqrt(len(x))
).reset_index()

print(anat_std_err_df)


# In[15]:


func_std_err_df = func_df.groupby('Visual Area')['Dice Score'].agg(
    mean='mean',
    std_error=lambda x: np.std(x, ddof=1) / np.sqrt(len(x))
).reset_index()

print(func_std_err_df)


# In[66]:


areas = ['hV4', 'VO1', 'VO2']
anat = anat_df[anat_df['Visual Area'].isin(areas)].copy()
func = func_df[func_df['Visual Area'].isin(areas)].copy()

anat['Input'] = 'Anatomical'
func['Input'] = 'Functional'

combined_df = pd.concat([anat, func], ignore_index=True)

anat_summary = anat.groupby('Visual Area')['Dice Score'].agg(mean='mean', std_error='sem').loc[areas].reset_index()
func_summary = func.groupby('Visual Area')['Dice Score'].agg(mean='mean', std_error='sem').loc[areas].reset_index()

plt.figure(figsize=(8, 6))
x_indices = np.arange(3)
shift = 0.2

sns.stripplot(
    data=combined_df, 
    x='Visual Area', 
    y='Dice Score', 
    hue='Input', 
    order=areas,
    dodge=True,
    alpha=0.2,
    palette=['#c5b4e3', '#ffc700'], 
    jitter=0.15,
    size=5
)

plt.errorbar(
    x=x_indices - shift, 
    y=anat_summary['mean'], 
    yerr=anat_summary['std_error'], 
    fmt='o', capsize=10, color='#4b2e83', markersize=5, elinewidth=2, markeredgecolor='black',
    label='_nolegend_'   # Prevents duplicate legend entries
)

plt.errorbar(
    x=x_indices + shift, 
    y=func_summary['mean'], 
    yerr=func_summary['std_error'], 
    fmt='s', capsize=10, color='#85754d', markersize=5, elinewidth=2, markeredgecolor='black',
    label='_nolegend_'
)

plt.xticks(ticks=x_indices, labels=areas)
plt.yticks(np.arange(0.0, 1.05, 0.05))
plt.ylabel('Dice Score')
plt.xlabel('Visual Area')
plt.title('Dice Scores by Visual Area with Mean ± Standard Error')
plt.grid(axis='y', linestyle='--', alpha=0.2)

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[0:2], labels[0:2], title="Inputs")

plt.tight_layout()
plt.show()


# In[75]:


fig, ax = plt.subplots(figsize=(8, 6))
x_indices = np.arange(3)

areas = ['hV4', 'VO1', 'VO2']
anat = anat_df[anat_df['Visual Area'].isin(areas)].copy()
anat_summary = anat.groupby('Visual Area')['Dice Score'].agg(mean='mean', std_error='sem').loc[areas].reset_index()

palette = {
    "R1": "#aadb1e",
    "R2": "#2ad2c9",
    "R3": "#e93cac",
    "R4": "#c5b4e3",
    "R5": "#ffc700",
}

sns.stripplot(
    data=anat,
    x='Visual Area',
    y= 'Dice Score',
    hue="Rater",
    palette=palette,
    dodge=True,
    order=areas,
    jitter=0.2,
    size=5,
    alpha=0.6,
    ax=ax
)


ax.errorbar(
    x=x_indices,
    y=anat_summary['mean'],
    yerr=anat_summary['std_error'],
    fmt="o",
    color="black",
    markersize=5,
    capsize=10,
    linewidth=2,
    label="Mean ± SEM"
)

plt.yticks(np.arange(0.0, 1.05, 0.05))
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.xticks(ticks=x_positions, labels=areas)
ax.set_xlabel("Visual Area")
ax.set_ylabel("Dice Score")
ax.set_title("Dice Score by Visual Area (Anatomical)")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.show()


# In[76]:


fig, ax = plt.subplots(figsize=(8, 6))
x_indices = np.arange(3)

areas = ['hV4', 'VO1', 'VO2']
func = func_df[func_df['Visual Area'].isin(areas)].copy()
func_summary = func.groupby('Visual Area')['Dice Score'].agg(mean='mean', std_error='sem').loc[areas].reset_index()

palette = {
    "R1": "#aadb1e",
    "R2": "#2ad2c9",
    "R3": "#e93cac",
    "R4": "#c5b4e3",
    "R5": "#ffc700",
}

sns.stripplot(
    data=func,
    x='Visual Area',
    y= 'Dice Score',
    hue="Rater",
    palette=palette,
    dodge=True,
    order=areas,
    jitter=0.2,
    size=5,
    alpha=0.6,
    ax=ax
)


ax.errorbar(
    x=x_indices,
    y=func_summary['mean'],
    yerr=func_summary['std_error'],
    fmt="o",
    color="black",
    markersize=5,
    capsize=10,
    linewidth=2,
    label="Mean ± SEM"
)

plt.yticks(np.arange(0.0, 1.05, 0.05))
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.xticks(ticks=x_positions, labels=areas)
ax.set_xlabel("Visual Area")
ax.set_ylabel("Dice Score")
ax.set_title("Dice Score by Visual Area (Functional)")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.show()

