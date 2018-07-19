''' Corrects the RMS data observed on 21-05-2018 temporally and spatially. 
'''

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import

import os
import DataAnalysisTools as dat

# Main directory
PRE_DIR = '/home/patrik/Dropbox/Workspace/'

# Directory where the raw data is stored
raw = PRE_DIR + 'RollingShutterSim/Observations/0521/CA0001_20180521_012018_461559/'

# Where the corrected data is stored
ftps_dir = PRE_DIR + 'RollingShutterSim/Observations/0521/ftps'

# FTPdetectinfo file
ftp_name = 'FTPdetectinfo_CA0001_20180521_012018_461559.txt'

# Where the source code is stored
rms_dir = PRE_DIR + 'RMS/'
project_dir = PRE_DIR + 'RollingShutterSim/'

# Needed for performing the corrections
img_y = 720

# Find all fits files in the raw directory
fits_files = dat.findFiles(raw, '.fits')

# Correct the data temporally and spatially
dat.correctDataSpatial(raw, ftp_name, fits_files, raw, ftp_name[:-4] + '_spatial.txt', img_y)
dat.correctCelestial(rms_dir, project_dir, raw, ftp_name[:-4] + '_spatial.txt')
dat.correctDataTemporal(raw, ftp_name, fits_files, raw, ftp_name[:-4] + '_temporal.txt', img_y, time_mark='middle')

# Copy the data to the ftps directory
os.system('cp ' + raw + ftp_name + ' ' + raw + ftp_name[:-4] + '_spatial.txt ' + raw + ftp_name[:-4] + '_temporal.txt ' + ftps_dir)