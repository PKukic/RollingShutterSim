''' Corrects the observed RMS data temporally and spatially; observed on 14-06-2018. 
'''

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import

import os
import numpy as np 

import DataAnalysisTools as dat

PRE_DIR = '/home/patrik/Dropbox/Workspace/'

# Directories where the raw files are stored
raw_a = PRE_DIR + 'RollingShutterSim/Observations/allRMS/CA0001_20180614_014042_812605_detected/'
raw_b = PRE_DIR + 'RollingShutterSim/Observations/allRMS/CA0003_20180614_024551_045135_detected/'
fits_dir = '../Observations/allRMS/'

# FTPdetectinfo files
ftp_name_a = 'FTPdetectinfo_CA0001_20180614_014042_812605.txt'
ftp_name_b = 'FTPdetectinfo_CA0003_20180614_024551_045135.txt'

# Where the source code is stored
rms_dir = PRE_DIR + 'RMS/'
project_dir = PRE_DIR + 'RollingShutterSim/'

# Location of the corrected files 
ftps_dir = '../Observations/allRMS/ftps/'

# Needed for coordinate corrections
img_y = 720

# Get array of FITS files
fits_files = np.genfromtxt(fits_dir + 'fits_files.txt', dtype='str')

# Correct the data from both CA001 and CA003 temporally and spatially
dat.correctDataSpatial(raw_a, ftp_name_a, fits_files, raw_a, ftp_name_a[:-4] + '_spatial.txt', img_y)
dat.correctCelestial(rms_dir, project_dir, raw_a, ftp_name_a[:-4] + '_spatial.txt')
dat.correctDataTemporal(raw_a, ftp_name_a, fits_files, raw_a, ftp_name_a[:-4] + '_temporal.txt', img_y, time_mark='middle')


dat.correctDataSpatial(raw_b, ftp_name_b, fits_files, raw_b, ftp_name_b[:-4] + '_spatial.txt', img_y)
dat.correctCelestial(rms_dir, project_dir, raw_b, ftp_name_b[:-4] + '_spatial.txt')
dat.correctDataTemporal(raw_b, ftp_name_b, fits_files, raw_b, ftp_name_b[:-4] + '_temporal.txt', img_y, time_mark='middle')


# Copy to corrected files directory
os.system('cp ' + raw_a + ftp_name_a[:-4] + '_spatial.txt ' + raw_a + ftp_name_a[:-4] + '_temporal.txt ' + ftps_dir)
os.system('cp ' + raw_b + ftp_name_b[:-4] + '_spatial.txt ' + raw_b + ftp_name_b[:-4] + '_temporal.txt ' + ftps_dir)