import DataAnalysisTools as dat 


raw_dir = '../Observations/0521/CA0001_20180521_012018_461559/'
ftp_dir = '../Observations/0521/ftps/'
ftp_name_spat = 'FTPdetectinfo_CA0001_20180521_012018_461559_spatial.txt'
ftp_name_temp = 'FTPdetectinfo_CA0001_20180521_012018_461559_temporal.txt'
ftp_name_nocorr = 'FTPdetectinfo_CA0001_20180521_012018_461559.txt'

fits_files = dat.findFITS(raw_dir)
save_dir = '../Observations/0521/ang_vel_RMS/'

dat.FTPtoAVT(ftp_dir, ftp_name_spat, fits_files, save_dir, corr_type = 'spat')
dat.FTPtoAVT(ftp_dir, ftp_name_temp, fits_files, save_dir, corr_type = 'temp')
dat.FTPtoAVT(ftp_dir, ftp_name_nocorr, fits_files, save_dir, corr_type = 'nocorr')