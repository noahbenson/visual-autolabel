# -*- coding: utf-8 -*-
################################################################################
# visual_autolabel/benson2025/_core.py


# Training Feature Sets.........................................................
# The base feature-sets we are predicting:
vaonly_properties = ('hV4', 'VO1', 'VO2')
econly_properties = ('E0', 'E1', 'E2', 'E3', 'E4')
# The base feature-sets we use to predict the above labels:
t1only_properties = ('x', 'y', 'z',
                     'curvature', 'convexity',
                     'thickness', 'surface_area')
fnonly_properties = ('prf_x', 'prf_y', 'prf_sigma', 'prf_cod')

# The OSF Repository............................................................
# This is used to download data and models.
osf_repository = 'osf://c49dv/'
