import numpy as np
import sys
import os

RMS_DIR = '/home/patrik/RMS/RMS/'
sys.path.insert(0, RMS_DIR)
from Formats import FTPdetectinfo as ftp

import SimulationTools as st

def thetaToH(theta):
	''' Converts the theta angle to height (elevation). 

		Arguments:
			theta: [float] Theta angle, measured from the zenith to the ground. 

		Return:
			height: [float] The height corresponding to a given theta angle. 
	'''

	height = 90 - theta

	return height


def phiToAz(phi):
	''' Converts the phi angle to azimuth. 

		Arguments:
			phi: [float] Phi angle. 

		Return:
			azim: [float] Azimuth. 
	'''

	azim = (90 - phi) % 360

	return azim


# Distance between two positions in the spherical coordinate system function
def angleDist(pos_a, pos_b):
	''' Calculates the distance between two points in a spherical coordinate system. 

		Arguments:
			pos_a: [tuple of floats] First position composed of (height, azimuth) coordinates. (deg, deg)
			pos_b: [tuple of floats] Second position composed of (height, azimuth) coordinates. (deg, deg)

		Return:
			dist: [tuple of floats] Distance between the first and second points. (deg)
	'''

	# Unpack angles from coordinate tuple and convert to  radians
	h_a = np.deg2rad(pos_a[0])
	azim_a = np.deg2rad(pos_a[1])

	h_b = np.deg2rad(pos_b[0])
	azim_b = np.deg2rad(pos_b[1])

	# Calculate distance
	dist = np.arccos(np.sin(h_a)*np.sin(h_b) + np.cos(h_a)*np.cos(h_b)*np.cos(azim_b - azim_a))

	# Convert to radians 
	dist = np.rad2deg(dist)

	return dist

def findFITS(fits_dir):

    fits_files = []

    for root, dirs, files in os.walk(fits_dir):
        for file in files:
            if file.endswith('.fits'):
                fits_files.append(file)

    return fits_files



def phiList(data_dir, data_name, fits_files):

    data = ftp.readFTPdetectinfo(data_dir, data_name)

    phi_arr = []

    for i in range(len(data)):

        meteor_data = data[i]
        ff_name, phi = meteor_data[0], meteor_data[10]

        if ff_name in fits_files:
            phi_arr.append(phi)

    return phi_arr


def correctDataSpatial(data_dir, data_name, fits_files, ftp_dir, ftp_name, img_y):

    data = ftp.readFTPdetectinfo(data_dir, data_name)

    meteor_list_spat = []

    for i in range(len(data)):

        meteor_data = data[i]

        ff_name, cam_code, meteor_no, n_segments, fps, hnr, mle, binn, pix_fm, rho, phi, meteor_meas = \
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

            # Length of all arrays
            n_coord = len(frame_n_arr)

            # Construct array of tuples of coordinates
            coord_arr = []
            coord_arr += [(col_arr[coord_i], row_arr[coord_i]) for coord_i in range(n_coord)]

            # Correct spatial coordinates
            coord_arr_corr = st.coordinateCorrection(t_arr, coord_arr, img_y, fps, version = 'v_corr')
            col_arr_corr = [x[0] for x in coord_arr_corr]
            row_arr_corr = [x[1] for x in coord_arr_corr]

            # Construct centroids array
            centroids_spat = np.c_[frame_n_arr, col_arr_corr, row_arr_corr, ra_arr, dec_arr, azim_arr, elev_arr, inten_arr, mag_arr]

            # Construct meteor 
            meteor_spat = [ff_name, meteor_no, rho, phi, centroids_spat]

            meteor_list_spat.append(meteor_spat)

    # Extract camera code and FPS
    cam_code = data[0][1]
    fps = data[0][4]

    ftp.writeFTPdetectinfo(meteor_list_spat, ftp_dir, ftp_name, ftp_dir, cam_code, fps, calibration=None, celestial_coords_given=True)

    return None


def correctDataTemporal(data_dir, data_name, fits_files, ftp_dir, ftp_name, img_y, time_mark):

    data = ftp.readFTPdetectinfo(data_dir, data_name)

    meteor_list_temp = []

    for i in range(len(data)):

        meteor_data = data[i]

        ff_name, cam_code, meteor_no, n_segments, fps, hnr, mle, binn, pix_fm, rho, phi, meteor_meas = \
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

            # Length of all arrays
            n_coord = len(frame_n_arr)

            # Construct array of tuples of coordinates
            coord_arr = []
            coord_arr += [(col_arr[coord_i], row_arr[coord_i]) for coord_i in range(n_coord)]

            # Correct temporal coordinates
            t_arr_corr = st.timeCorrection(coord_arr, img_y, fps, t_init = t_arr[0], time_mark = time_mark)
            frame_n_arr_corr = [x*fps for x in t_arr_corr]

            # Construct centroids array
            centroids_temp = np.c_[frame_n_arr_corr, col_arr, row_arr, ra_arr, dec_arr, azim_arr, elev_arr, inten_arr, mag_arr]

            # Construct meteor 
            meteor_temp = [ff_name, meteor_no, rho, phi, centroids_temp]

            meteor_list_temp.append(meteor_temp)

    # Extract camera code and FPS
    cam_code = data[0][1]
    fps = data[0][4]

    ftp.writeFTPdetectinfo(meteor_list_temp, ftp_dir, ftp_name, ftp_dir, cam_code, fps, calibration=None, celestial_coords_given=True)

    return None


def correctCelestial(rms_dir, project_dir, raw_dir, ftp_name):

    os.chdir(rms_dir)
    os.system('python -m RMS.Astrometry.ApplyAstrometry ' + raw_dir + ftp_name)
    os.chdir(project_dir + 'Source/')

    return None

def FTPtoAVT(data_dir, data_name, fits_files):

    data = ftp.readFTPdetectinfo(data_dir, data_name)

    for i in range(len(data)):

        meteor_data = data[i]

        ff_name = meteor_data[0]
        fps = meteor_data[4]
        meteor_meas = meteor_data[11]

        if ff_name in fits_files:
            
            # Time and position arrays
            t_arr = [x[1]/fps for x in meteor_meas]
            ra_arr = [x[4] for x in meteor_meas]
            dec_arr = [x[5] for x in meteor_meas]

            # Length of all arrays
            n_coord = len(t_arr)

            dist_arr = []
            deltat_arr = []
            av_arr = []
            fint_arr = []

            # Construct delta time and delta distance arrays
            for coord_i in range(1, n_coord):
                pos_a = (dec_arr[coord_i-1], ra_arr[coord_i-1])
                pos_b = (dec_arr[coord_i], ra_arr[coord_i])

                dist = angleDist(pos_a, pos_b)
                dist_arr.append(dist)

                deltat = t_arr[coord_i] - t_arr[coord_i-1]
                deltat_arr.append(deltat)

            for coord_i in range(n_coord-1):

                av = dist_arr[coord_i]/deltat_arr[coord_i]
                av_arr.append(av)

                fint_arr.append(t_arr[coord_i+1])

            # Initial time coordinate and velocity
            fint_arr.insert(0, t_arr[0])
            av_arr.insert(0, av_arr[0])

    return fint_arr, av_arr