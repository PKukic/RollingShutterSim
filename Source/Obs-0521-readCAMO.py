''' Reads out the CAMO data observed on 05-21-2018 and stores the centroid angular velocities.
'''

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import

import DataAnalysisTools as dat 

# Where the data is stored
data_dir = '../Observations/0521/camo_20180521/'
data_name_arr = dat.findFiles(data_dir, '.txt')

# Target directory of the angular velocity arrays
save_dir = '../Observations/0521/ang_vel_CAMO/'

# Go through each event, and store the appropriate AV arrays as NPZ files
for name in data_name_arr:
	dat.CAMOtoAVT(data_dir, name, save_dir)