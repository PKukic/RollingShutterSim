from Functions import *
import numpy as np

raw_a = '/home/patrik/Workspace/RollingComp/Data/allRMS/CA0001_20180614_014042_812605_detected/'
raw_b = '/home/patrik/Workspace/RollingComp/Data/allRMS/CA0003_20180614_024551_045135_detected/'

ftp_name_a = 'FTPdetectinfo_CA0001_20180614_014042_812605.txt'
ftp_name_b = 'FTPdetectinfo_CA0003_20180614_024551_045135.txt'

rms_dir = '/home/patrik/RMS/'
project_dir = '/home/patrik/Workspace/RollingComp/'
ftps_dir = '../Data/allRMS/ftps/'

img_y = 720

fits_files = np.genfromtxt('../Data/allRMS/fits_files.txt', dtype='str')


correctDataSpatial(raw_a, ftp_name_a, fits_files, raw_a, ftp_name_a[:-4] + '_spatial.txt', img_y)
correctCelestial(rms_dir, project_dir, raw_a, ftp_name_a[:-4] + '_spatial.txt')
correctDataTemporal(raw_a, ftp_name_a, fits_files, raw_a, ftp_name_a[:-4] + '_temporal.txt', img_y, time_mark='middle')


correctDataSpatial(raw_b, ftp_name_b, fits_files, raw_b, ftp_name_b[:-4] + '_spatial.txt', img_y)
correctCelestial(rms_dir, project_dir, raw_b, ftp_name_b[:-4] + '_spatial.txt')
correctDataTemporal(raw_b, ftp_name_b, fits_files, raw_b, ftp_name_b[:-4] + '_temporal.txt', img_y, time_mark='middle')


os.system('cp ' + raw_a + ftp_name_a[:-4] + '_spatial.txt ' + raw_a + ftp_name_a[:-4] + '_temporal.txt ' + ftps_dir)
os.system('cp ' + raw_b + ftp_name_b[:-4] + '_spatial.txt ' + raw_b + ftp_name_b[:-4] + '_temporal.txt ' + ftps_dir)