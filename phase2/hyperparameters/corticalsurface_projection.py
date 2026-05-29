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
from visual_autolabel.benson2025.hcp import HCPDataset


# In[ ]:


ny.config["freesurfer_subject_paths"] =  "/gscratch/nbenlab/data/freesurfer/subjects"
ny.config["data_cache_root"] = "/gscratch/nbenlab/data/neuropythy"
ny.config["hcp_subject_paths"] = "/gscratch/nbenlab/data/hcp/subjects"
ny.config["hcp_auto_download"] = True
ny.config["hcp_lines_path"] = "/gscratch/nbenlab/data/hcp/lines"
ny.config["hcp_metadata_path"] = "/gscratch/nbenlab/data/hcp/meta"
ny.config["visual_performance_fields_path"] = "/gscratch/nbenlab/data/performance-fields"


# In[ ]:


# PATH CONFIGURATION
data_root = '/gscratch/nbenlab/data/visual-autolabel' # CHECK TO SEE HOW YOU WANT TO --bind THE FILES AND CHANGE THIS ACCORDINGLY.
dataset_cache_path = f'{data_root}/datasets/'
model_cache_path = f'{data_root}/models'
figures_path = f'{data_root}/figures'


# In[ ]:


# MORE PATH CONFIGURATION
project.config.dataset_cache_path = dataset_cache_path
project.config.model_cache_path = model_cache_path
project.config.figures_path = figures_path


# In[ ]:


# NB: NOT THE VALIDATION SET WE WANT TO SUE
val = np.array([814649, 381038, 116726, 100610, 770352, 971160, 541943, 134829,
        585256, 131217, 209228, 926862, 146129, 135124, 927359, 212419,
        825048, 257845, 318637, 898176, 389357, 654552, 536647, 173334,
        180533, 155938, 547046, 191033, 346137, 395756, 196144, 137128,
        627549])


# In[ ]:


# hem = 'lh'
# for sid in val:
#     labels_gold = ny.load(f'/gscratch/nbenlab/data/hcp/labels/visual/{hem}.{sid}.mgz')
#     print(len(labels_gold))


# In[ ]:


sid = 182739
hem = 'lh'

labels_gold = ny.load(f'/gscratch/nbenlab/data/hcp/labels/visual/{hem}.{sid}.mgz')


# In[ ]:


weights_file = f'{model_cache_path}/hV4_VO1_VO2_optuna_func_0/best_inputs.pt'


# In[ ]:


weights = torch.load(weights_file, weights_only=True, map_location=torch.device('cpu'))
model = va.UNet(feature_count=11, segment_count=3)
model.load_state_dict(weights)


# In[ ]:


im = HCPDataset(
    inputs='func',
    outputs='area',
    sids=val,
    cache_path = os.path.join(dataset_cache_path, 'HCP'),
    hemis='lr'
)


# In[ ]:


sid = 111312
ny.config = "/gscratch/nbenlab/data/hcp/subjects"
sub = ny.data['hcp_lines'].subjects[sid]


# In[ ]:


labels = im.predlabels(k=0, model=model, labelsets={'visual_area': slice(3, 7)})


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# FIGURING OUT HOW TO FIX THE KEY ERROR ISSUE & LEARNING TO WORK WITH THE _to_target_index function. 


# In[ ]:


raters = ['A7', 'A8', 'A11', 'A12', 'A13']
val = np.array([115017, 126426, 134829, 146735, 169444, 171633, 172130, 176542,
        177140, 182739, 185442, 192641, 195041, 198653, 204521, 209228,
        212419, 239136, 246133, 263436, 320826, 360030, 380036, 389357,
        581450, 585256, 601127, 765864, 782561, 789373, 825048, 871762,
        901442])

for rater in raters:
    for sid in val:
        try: 
            target, index = im._to_target_index({'rater': rater, 'subject': sid})
            print(f'Rater {rater}, SID {sid} has index: {index}')
            labels = im.predlabels(k=index, model=model, labelsets={'visual_area': slice(3, 7)})
        except Exception as e:
            print(f'Error for rater {rater} and subject {sid}: {e}')


# In[ ]:





# In[ ]:


ny.data['hcp_lines'].subject_labels['A1'][199655]['rh']


# In[ ]:


# FIGURING OUT HOW THE FUNCTION WORKS WITH FOR 181 NUMBERS
for num in range(0, 181):
    try:
        print(im._to_target_index(num))
    except Exception as e:
        print(f"Error on number {num}: {e}")


# In[ ]:


# REVERSE ENGINEERING TO EXTRACT INDEX FROM THE _to_target_index() function
target, index = im._to_target_index({'rater': 'A8', 'subject': 146735})
print(index)


# In[ ]:


# SEEING IF ALL OF THE VALIDATION SIDs WORK FOR THE FUNCTION
raters = ['A7', 'A8', 'A11', 'A12', 'A13']
val = np.array([115017, 126426, 134829, 146735, 169444, 171633, 172130, 176542,
        177140, 182739, 185442, 192641, 195041, 198653, 204521, 209228,
        212419, 239136, 246133, 263436, 320826, 360030, 380036, 389357,
        581450, 585256, 601127, 765864, 782561, 789373, 825048, 871762,
        901442])

for rater in raters:
    for sid in val:
        try: 
            target, index = im._to_target_index({'rater': rater, 'subject': sid})
            print(f'Rater {rater}, SID {sid} has index: {index}')
        except Exception as e:
            print(f'Error for rater {rater} and subject {sid}: {e}')

