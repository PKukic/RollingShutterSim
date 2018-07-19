import numpy as np
import os
import sys

import SimulationTools as st 
import DataAnalysisTools as dat

PRE_DIR = '/home/patrik/Dropbox/Workspace/'

# Directory in which the RMS codes are saved
RMS_DIR = PRE_DIR + 'RMS/'

sys.path.insert(0, RMS_DIR + 'RMS/')
from Formats import FTPdetectinfo as ftp

# Directory where the project and the source code itself are stored
PROJECT_DIR = PRE_DIR + 'RollingShutterSim/' 
SOURCE_DIR = PROJECT_DIR +  'Source/'

# Raw data directory
RAW_DIR = '../Observations/0521/CA0001_20180521_012018_461559/raw/'

# FTPdetectinfo file name
FTP_FILE = 'FTPdetectinfo_CA0001_20180521_012018_461559'

# platepar file name
PLATEPAR_FILE = 'platepar_cmn2010.cal'

# Directories where both the angular velocity data and the corrected angular velocity data are saved
AV_NOCORR_DIR =  '../Observations/0521/CA0001_20180521_012018_461559/ang_vel_nocorr/'
AV_TEMP_DIR = '../Observations/0521/CA0001_20180521_012018_461559/ang_vel_temp/'
AV_SPAT_DIR = '../Observations/0521/CA0001_20180521_012018_461559/ang_vel_spat/'

# Where the headings of meteors across the image are stored
ANG_DIR = '../Observations/0521/angles/'

# IMAGE RESOLUTION
img_x = 1280
img_y = 720


# Get all files in the raw data directory that end with *.fits
fits_files = []

for root, dirs, files in os.walk(RAW_DIR):
	for file in files:
		if file.endswith('.fits'):
			fits_files.append(file)

# print(fits_files)

# Read FTPdetectinfo
data = ftp.readFTPdetectinfo(RAW_DIR, FTP_FILE + '.txt')

meteor_list_nocorr = []
meteor_list_temp = []
meteor_list_spat = []

phi_arr = []

# Go through every meteor
for i in range(len(data)):

	meteor_data = data[i]

	# Unpack meteor data
	ff_name, cam_code, meteor_no, n_segments, fps, hnr, mle, binn, px_fm, rho, phi, meteor_meas = \
		meteor_data
	
	if ff_name in fits_files:
		
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


		# Length of coordinates array
		n_coord = len(t_arr)

		# Corrected coordinates
		coord_arr = []
		t_arr_corr = []

		# print(frame_n_arr)
		# print(t_arr)

		# Pack coordinates into list of tuples
		for coord_i in range(n_coord):
			coord_arr.append((col_arr[coord_i], row_arr[coord_i]))

		# Correct time coordinates
		t_arr_corr = st.timeCorrection(coord_arr, img_y, fps, 0, time_mark = 'middle', t_init = t_arr[0])
		frame_n_arr_corr = [x*fps for x in t_arr_corr]

		print(frame_n_arr_corr) 

		# Correct spatial coordinates
		coord_arr_corr = st.coordinateCorrection(t_arr, coord_arr, img_y, fps, version = 'v_corr')
		col_arr_corr = [x[0] for x in coord_arr_corr]
		row_arr_corr = [x[1] for x in coord_arr_corr]

		# print(col_arr)
		# print(col_arr_corr)
		# print(row_arr)
		# print(row_arr_corr)
		
		# Construct centroids arrays
		centroids_nocorr = np.c_[frame_n_arr, col_arr, row_arr, ra_arr, dec_arr, azim_arr, elev_arr, inten_arr, mag_arr]
		centroids_temp = np.c_[frame_n_arr_corr, col_arr, row_arr, ra_arr, dec_arr, azim_arr, elev_arr, inten_arr, mag_arr]
		centroids_spat = np.c_[frame_n_arr, col_arr_corr, row_arr_corr, ra_arr, dec_arr, azim_arr, elev_arr, inten_arr, mag_arr]
	
		meteor_nocorr = [ff_name, meteor_no, rho, phi, centroids_nocorr]
		meteor_temp = [ff_name, meteor_no, rho, phi, centroids_temp]
		meteor_spat = [ff_name, meteor_no, rho, phi, centroids_spat]


		meteor_list_nocorr.append(meteor_nocorr)
		meteor_list_temp.append(meteor_temp)
		meteor_list_spat.append(meteor_spat)


		phi_arr.append(phi)

# print('#'*20)
# print(len(phi_arr))
# print('#'*20)


# Define camera code and FPS
cam_code = data[0][1]
fps = data[0][4]

ftp.writeFTPdetectinfo(meteor_list_nocorr, RAW_DIR, FTP_FILE + '_nocorr.txt', RAW_DIR, cam_code, fps, calibration=None, celestial_coords_given=True)
ftp.writeFTPdetectinfo(meteor_list_temp, RAW_DIR, FTP_FILE + '_temp.txt', RAW_DIR, cam_code, fps, calibration=None, celestial_coords_given=True)
ftp.writeFTPdetectinfo(meteor_list_spat, RAW_DIR, FTP_FILE + '_spat.txt', RAW_DIR, cam_code, fps, calibration=None, celestial_coords_given=True)
np.savez(ANG_DIR + 'rms.npz', phi_arr)

#################################################################


# Determine the celestial coordinates of meteors which are spatially corrected
os.chdir(RMS_DIR)
os.system('python -m RMS.Astrometry.ApplyAstrometry ' + PROJECT_DIR + RAW_DIR[3:] + FTP_FILE + '_spat.txt')
os.chdir(SOURCE_DIR)

# Extract data
data_nocorr = ftp.readFTPdetectinfo(RAW_DIR, FTP_FILE + '_nocorr.txt')
data_temp = ftp.readFTPdetectinfo(RAW_DIR, FTP_FILE + '_temp.txt')
data_spat = ftp.readFTPdetectinfo(RAW_DIR, FTP_FILE + '_spat.txt')

# print(data_temp)

for i in range(len(data_nocorr)):

	ff_name = data_nocorr[i][0]
	fps = data_nocorr[i][4]
	
	meteor_meas_nocorr, meteor_meas_temp, meteor_meas_spat = data_nocorr[i][11], data_temp[i][11], data_spat[i][11]

	filename = ff_name[:-5]

	print(filename, fps)
	print(ff_name)

	print(data_nocorr[i][0], data_temp[i][0], data_spat[i][0])


	if ff_name in fits_files:

		nframe_arr_nocorr = [x[1] for x in meteor_meas_nocorr]
		t_arr_nocorr = [x/fps for x in nframe_arr_nocorr]

		nframe_arr_temp = [x[1] for x in meteor_meas_temp]
		t_arr_temp = [x/fps for x in nframe_arr_temp]

		nframe_arr_spat = [x[1] for x in meteor_meas_spat]
		t_arr_spat = [x/fps for x in nframe_arr_spat]


		print(len(nframe_arr_nocorr), len(nframe_arr_temp))

		ra_arr_nocorr = [x[4] for x in meteor_meas_nocorr]
		dec_arr_nocorr = [x[5] for x in meteor_meas_nocorr]

		ra_arr_temp = [x[4] for x in meteor_meas_temp]
		dec_arr_temp = [x[5] for x in meteor_meas_temp]
		
		ra_arr_spat = [x[4] for x in meteor_meas_spat]
		dec_arr_spat = [x[5] for x in meteor_meas_spat]

		n_coord = len(nframe_arr_temp)

		dist_n_arr = []
		dist_t_arr = []
		dist_s_arr = []

		deltat_n_arr = []
		deltat_t_arr = []
		deltat_s_arr = []

		av_n_arr = []
		av_t_arr = []
		av_s_arr = []

		fint_n_arr = []
		fint_t_arr = []
		fint_s_arr = []

		for coord_i in range(1, n_coord):

			pos_a_n = (dec_arr_nocorr[coord_i-1], ra_arr_nocorr[coord_i-1])
			pos_b_n = (dec_arr_nocorr[coord_i], ra_arr_nocorr[coord_i])

			pos_a_t = (dec_arr_temp[coord_i-1], ra_arr_temp[coord_i-1])
			pos_b_t = (dec_arr_temp[coord_i], ra_arr_temp[coord_i])

			pos_a_s = (dec_arr_spat[coord_i-1], ra_arr_spat[coord_i-1])
			pos_b_s = (dec_arr_spat[coord_i], ra_arr_spat[coord_i])


			dist_n = dat.angleDist(pos_a_n, pos_b_n)
			dist_n_arr.append(dist_n)

			dist_t = dat.angleDist(pos_a_t, pos_b_t)
			dist_t_arr.append(dist_t)

			dist_s = dat.angleDist(pos_a_s, pos_b_s)
			dist_s_arr.append(dist_s)


			deltat_n = t_arr_nocorr[coord_i] - t_arr_nocorr[coord_i-1]
			deltat_t = t_arr_temp[coord_i] - t_arr_temp[coord_i-1]
			deltat_s = t_arr_spat[coord_i] - t_arr_spat[coord_i-1]

			deltat_n_arr.append(deltat_n)
			deltat_t_arr.append(deltat_t)
			deltat_s_arr.append(deltat_s)


		for coord_i in range(n_coord-1):

			# print(coord_i)

			av_n_arr.append(dist_n_arr[coord_i]/deltat_n_arr[coord_i])
			av_t_arr.append(dist_t_arr[coord_i]/deltat_t_arr[coord_i])
			av_s_arr.append(dist_s_arr[coord_i]/deltat_s_arr[coord_i])
		
			fint_n_arr.append(t_arr_nocorr[coord_i+1])
			fint_t_arr.append(t_arr_temp[coord_i+1])
			fint_s_arr.append(t_arr_spat[coord_i+1])


		# Assume that the initial velocity is equal to its subsequent velocity
		av_n_arr.insert(0, av_n_arr[0])
		av_t_arr.insert(0, av_t_arr[0])
		av_s_arr.insert(0, av_s_arr[0])

		# Assume that the initial time is equal to the next time
		fint_n_arr.insert(0, fint_n_arr[0])
		fint_t_arr.insert(0, fint_t_arr[0])
		fint_s_arr.insert(0, fint_s_arr[0])

		for i in range(n_coord):
			print('{:.2f}, {:.2f}; {:.2f}, {:.2f}; {:.2f}, {:.2f}'.format(fint_n_arr[i], av_n_arr[i], fint_t_arr[i], av_t_arr[i], fint_s_arr[i], av_s_arr[i]))


		# Save the files
		np.savez(AV_NOCORR_DIR + filename + '_nocorr.npz', *[fint_n_arr, av_n_arr])
		np.savez(AV_TEMP_DIR + filename + '_temp.npz', *[fint_t_arr, av_t_arr])
		np.savez(AV_SPAT_DIR + filename + '_spat.npz', *[fint_s_arr, av_s_arr])