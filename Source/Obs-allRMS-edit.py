''' Merges the FTPdetectinfo files from stations CA001 and CA003 for 14-06-2018 into one file. Separate files for temporally and spatially corrected data.
'''
# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import

import numpy as np
import sys

RMS_DIR = '/home/patrik/Dropbox/Workspace/RMS/RMS/'
sys.path.insert(0, RMS_DIR)
from Formats import FTPdetectinfo as ftp

ftps_dir = '../Observations/allRMS/ftps/'

# Read the FTP data
data_temp_o = ftp.readFTPdetectinfo(ftps_dir, 'FTPdetectinfo_CA0001_20180614_014042_812605_temporal.txt')
data_temp_t = ftp.readFTPdetectinfo(ftps_dir, 'FTPdetectinfo_CA0003_20180614_024551_045135_temporal.txt')

data_spat_o = ftp.readFTPdetectinfo(ftps_dir, 'FTPdetectinfo_CA0001_20180614_014042_812605_spatial.txt')
data_spat_t = ftp.readFTPdetectinfo(ftps_dir, 'FTPdetectinfo_CA0003_20180614_024551_045135_spatial.txt')

meteor_list_spat = []
meteor_list_temp = []

for i in range(len(data_temp_o)):

	# FIRST TEMPORAL 
    meteor_temp_o = data_temp_o[i]

    ff_name, cam_code, meteor_no, n_segments, fps, hnr, mle, binn, pix_fm, rho, phi, meteor_meas = \
        meteor_temp_o

    # Unpack meteor measurements arrays
    frame_n_arr = [x[1] for x in meteor_meas]
    t_arr = [x/fps for x in frame_n_arr]
    col_arr = [x[2] for x in meteor_meas]
    row_arr = [x[3] for x in meteor_meas]
    ra_arr = [x[4] for x in meteor_meas]
    dec_arr = [x[5] for x in meteor_meas]
    azim_arr = [x[6] for x in meteor_meas]
    elev_arr = [x[7] for x in meteor_meas]
    inten_arr = [x[8] for x in meteor_meas]
    mag_arr = [x[9] for x in meteor_meas]

    # Construct centroids array
    centroids_temp_o = np.c_[frame_n_arr, col_arr, row_arr, ra_arr, dec_arr, azim_arr, elev_arr, inten_arr, mag_arr]

    # Construct meteor 
    meteor_temp_o = [ff_name, meteor_no, rho, phi, centroids_temp_o]

    meteor_list_temp.append(meteor_temp_o)

    # THIRD TEMPORAL
    meteor_temp_t = data_temp_t[i]

    ff_name, cam_code, meteor_no, n_segments, fps, hnr, mle, binn, pix_fm, rho, phi, meteor_meas = \
        meteor_temp_t

    # Unpack meteor measurements arrays
    frame_n_arr = [x[1] for x in meteor_meas]
    t_arr = [x/fps for x in frame_n_arr]
    col_arr = [x[2] for x in meteor_meas]
    row_arr = [x[3] for x in meteor_meas]
    ra_arr = [x[4] for x in meteor_meas]
    dec_arr = [x[5] for x in meteor_meas]
    azim_arr = [x[6] for x in meteor_meas]
    elev_arr = [x[7] for x in meteor_meas]
    inten_arr = [x[8] for x in meteor_meas]
    mag_arr = [x[9] for x in meteor_meas]

    # Construct centroids array
    centroids_temp_t = np.c_[frame_n_arr, col_arr, row_arr, ra_arr, dec_arr, azim_arr, elev_arr, inten_arr, mag_arr]

    # Construct meteor 
    meteor_temp_t = [ff_name, meteor_no, rho, phi, centroids_temp_t]

    meteor_list_temp.append(meteor_temp_t)

    # FIRST SPATIAL
    meteor_spat_o = data_spat_o[i]

    ff_name, cam_code, meteor_no, n_segments, fps, hnr, mle, binn, pix_fm, rho, phi, meteor_meas = \
        meteor_spat_o

    # Unpack meteor measurements arrays
    frame_n_arr = [x[1] for x in meteor_meas]
    t_arr = [x/fps for x in frame_n_arr]
    col_arr = [x[2] for x in meteor_meas]
    row_arr = [x[3] for x in meteor_meas]
    ra_arr = [x[4] for x in meteor_meas]
    dec_arr = [x[5] for x in meteor_meas]
    azim_arr = [x[6] for x in meteor_meas]
    elev_arr = [x[7] for x in meteor_meas]
    inten_arr = [x[8] for x in meteor_meas]
    mag_arr = [x[9] for x in meteor_meas]

    # Construct centroids array
    centroids_spat_o = np.c_[frame_n_arr, col_arr, row_arr, ra_arr, dec_arr, azim_arr, elev_arr, inten_arr, mag_arr]

    # Construct meteor 
    meteor_spat_o = [ff_name, meteor_no, rho, phi, centroids_spat_o]

    meteor_list_spat.append(meteor_spat_o)


    # THIRD SPATIAL
    meteor_spat_t = data_spat_t[i]

    ff_name, cam_code, meteor_no, n_segments, fps, hnr, mle, binn, pix_fm, rho, phi, meteor_meas = \
        meteor_spat_t

    # Unpack meteor measurements arrays
    frame_n_arr = [x[1] for x in meteor_meas]
    t_arr = [x/fps for x in frame_n_arr]
    col_arr = [x[2] for x in meteor_meas]
    row_arr = [x[3] for x in meteor_meas]
    ra_arr = [x[4] for x in meteor_meas]
    dec_arr = [x[5] for x in meteor_meas]
    azim_arr = [x[6] for x in meteor_meas]
    elev_arr = [x[7] for x in meteor_meas]
    inten_arr = [x[8] for x in meteor_meas]
    mag_arr = [x[9] for x in meteor_meas]

    # Construct centroids array
    centroids_spat_t = np.c_[frame_n_arr, col_arr, row_arr, ra_arr, dec_arr, azim_arr, elev_arr, inten_arr, mag_arr]

    # Construct meteor 
    meteor_spat_t = [ff_name, meteor_no, rho, phi, centroids_spat_t]

    meteor_list_spat.append(meteor_spat_t)

# Extract camera code and FPS
cam_code = data_temp_o[0][1]
fps = data_temp_o[0][4]

# Save the data as separate temporal and spatial FTPdetectinfo files
ftp.writeFTPdetectinfo(meteor_list_temp, ftps_dir, 'FTPdetectinfo_temporal.txt', ftps_dir, cam_code, fps, calibration=None, celestial_coords_given=True)
ftp.writeFTPdetectinfo(meteor_list_spat, ftps_dir, 'FTPdetectinfo_spatial.txt', ftps_dir, cam_code, fps, calibration=None, celestial_coords_given=True)