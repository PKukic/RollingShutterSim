import os
import numpy as np 

import DataAnalysisTools as dat

PRE_DIR = '/home/patrik/Dropbox/Workspace/'

raw_a = PRE_DIR + 'RollingShutterSim/Observations/allRMS/CA0001_20180614_014042_812605_detected/'
raw_b = PRE_DIR + 'RollingShutterSim/Observations/allRMS/CA0003_20180614_024551_045135_detected/'

ftp_name_a = 'FTPdetectinfo_CA0001_20180614_014042_812605.txt'
ftp_name_b = 'FTPdetectinfo_CA0003_20180614_024551_045135.txt'

rms_dir = PRE_DIR + 'RMS/'
project_dir = PRE_DIR + 'RollingShutterSim/'
ftps_dir = '../Observations/allRMS/ftps/'
fits_dir = '../Observations/allRMS/'

img_y = 720

fits_files = np.genfromtxt(fits_dir + 'fits_files.txt', dtype='str')


dat.correctDataSpatial(raw_a, ftp_name_a, fits_files, raw_a, ftp_name_a[:-4] + '_spatial.txt', img_y)
dat.correctCelestial(rms_dir, project_dir, raw_a, ftp_name_a[:-4] + '_spatial.txt')
dat.correctDataTemporal(raw_a, ftp_name_a, fits_files, raw_a, ftp_name_a[:-4] + '_temporal.txt', img_y, time_mark='middle')


dat.correctDataSpatial(raw_b, ftp_name_b, fits_files, raw_b, ftp_name_b[:-4] + '_spatial.txt', img_y)
dat.correctCelestial(rms_dir, project_dir, raw_b, ftp_name_b[:-4] + '_spatial.txt')
dat.correctDataTemporal(raw_b, ftp_name_b, fits_files, raw_b, ftp_name_b[:-4] + '_temporal.txt', img_y, time_mark='middle')


os.system('cp ' + raw_a + ftp_name_a[:-4] + '_spatial.txt ' + raw_a + ftp_name_a[:-4] + '_temporal.txt ' + ftps_dir)
os.system('cp ' + raw_b + ftp_name_b[:-4] + '_spatial.txt ' + raw_b + ftp_name_b[:-4] + '_temporal.txt ' + ftps_dir)