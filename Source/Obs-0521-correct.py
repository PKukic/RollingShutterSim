import os
import DataAnalysisTools as dat

PRE_DIR = '/home/patrik/Dropbox/Workspace/'

raw = PRE_DIR + 'RollingShutterSim/Observations/0521/CA0001_20180521_012018_461559/'

ftp_name = 'FTPdetectinfo_CA0001_20180521_012018_461559.txt'

rms_dir = PRE_DIR + 'RMS/'
project_dir = PRE_DIR + 'RollingShutterSim/'
ftps_dir = PRE_DIR + 'RollingShutterSim/Observations/0521/ftps'

img_y = 720

fits_files = dat.findFITS(raw)


dat.correctDataSpatial(raw, ftp_name, fits_files, raw, ftp_name[:-4] + '_spatial.txt', img_y)
dat.correctCelestial(rms_dir, project_dir, raw, ftp_name[:-4] + '_spatial.txt')
dat.correctDataTemporal(raw, ftp_name, fits_files, raw, ftp_name[:-4] + '_temporal.txt', img_y, time_mark='middle')


os.system('cp ' + raw + ftp_name + ' ' + raw + ftp_name[:-4] + '_spatial.txt ' + raw + ftp_name[:-4] + '_temporal.txt ' + ftps_dir)