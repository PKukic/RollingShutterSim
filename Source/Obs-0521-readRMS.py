''' Reads out the corrected RMS data observed on 05-21-2018 and stores the centroid angular velocities.
'''

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import

import DataAnalysisTools as dat 

# Where the raw data is stored
raw_dir = '../Observations/0521/CA0001_20180521_012018_461559/'
ftp_dir = '../Observations/0521/ftps/'

# Names of temporally and spatially corrected centroid data / uncorrected data
ftp_name_spat = 'FTPdetectinfo_CA0001_20180521_012018_461559_spatial.txt'
ftp_name_temp = 'FTPdetectinfo_CA0001_20180521_012018_461559_temporal.txt'
ftp_name_nocorr = 'FTPdetectinfo_CA0001_20180521_012018_461559.txt'

angles_dir = '../Observations/0521/angles/'

# Get all FITS
fits_files = dat.findFiles(raw_dir, '.fits')

save_dir = '../Observations/0521/ang_vel_RMS/'


# Read out the angular velocities and save them as NPZ files to save_dir
dat.FTPtoAVT(ftp_dir, ftp_name_spat, fits_files, save_dir, corr_type = 'spat')
dat.FTPtoAVT(ftp_dir, ftp_name_temp, fits_files, save_dir, corr_type = 'temp')
dat.FTPtoAVT(ftp_dir, ftp_name_nocorr, fits_files, save_dir, corr_type = 'nocorr')

# Save the meteor angles as a NPZ file
dat.saveAngles(ftp_dir, ftp_name_nocorr, fits_files, angles_dir)

