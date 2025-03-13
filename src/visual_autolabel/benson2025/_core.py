# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/benson2025/_core.py


# Training Feature Sets.........................................................
# The base feature-sets we are predicting:
caonly_properties = ('V1', 'V2', 'V3')
econly_properties = ('E0', 'E1', 'E2', 'E3', 'E4')
# The base feature-sets we use to predict the above labels:
t1only_properties = ('x', 'y', 'z',
                     'curvature', 'convexity',
                     'thickness', 'surface_area')
fnonly_properties = ('prf_x', 'prf_y', 'prf_sigma', 'prf_cod')
vaonly_properties = ('hV4', 'VO1', 'VO2')
daonly_properties = ('V3a', 'V3b', 'IPS0', 'LO1')

visual_area_label_key = {
    'V1': 1,
    'V2': 2,
    'V3': 3,
    'hV4': 4,
    'VO1': 5,
    'VO2': 6,
    'V3a': 7,
    'V3b': 8,
    'IPS0': 9,
    'LO1': 10}


visual_area_neighbors={
        'V1' :{'V2'},
        'V2' :{'V1','V3'},
        'V3' :{'V2','V3a','LO1','hV4'},
        'hV4':{'V3','VO1','LO1','LO2','TO1'},
        'VO1':{'hV4','VO2'},
        'VO2':{'VO1'},
        'V3a':{'V3','V3b','LO1','IPS0'},
        'V3b':{'V3a','LO1','IPS0'},
        'IPS0':{'V3a','V3b'},
        'LO1':{'V3a','V3b','LO1','LO2','hV4'},
        'LO2':{'LO1','hV4','TO1'},
        'TO1':{'LO2','hV4','TO2'},
        'TO2':{'TO1'},
},

visual_area_to_lines={
        'V1' :{'V1'},
        'V2' :{'V1','V2'},
        'V3' :{'V2','V3'},
        'hV4':{'V3','hV4'},
        'LO1':{'V3','hV4','LO1'},
        'V3a':{'V3','V3a'},
        'V3b':{'V3a','LO1','V3b'},
        'IPS0':{'V3a','V3b','IPS0'},
        'VO1':{'hV4','VO1'},
        'VO2':{'VO2','VO2'}
},


visual_area_label_names = tuple(
    map(
        lambda u: u[0],
        sorted(
            visual_area_label_key.items(),
            key=lambda u: u[1])))

input_descriptions = {
    'null': 'Nothing',
    'anat': 'T1 Only',
    't1t2': 'T1 & T2',
    'trac': 'T1 & DWI',
    'nofn': 'T1, T2, & DWI',
    'func': 'T1 & Retinotopy',
    'nodw': 'T1, T2, & Retinotopy',
    'not2': 'T1, DWI, & Retinotopy',
    'full': 'T1, T2, DWI, & Retinotopy',
    'tmpl': 'Benson et al. (2014), using T1',
    'warp': 'Benson & Winawer (2018), using T1 & Retinotopy',
    'rely': 'Inter-rater Reliability'}
output_descriptions = {
    'area': 'Visual Area Labels',
    'ring': 'Visual Ring Labels',
    'sect': 'Visual Sector Labels'}
