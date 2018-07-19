''' Tools for dealing with RMS and CAMO data, and applying coordinate corrections to the data.
'''

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import

import numpy as np
import sys
import os

RMS_DIR = '/home/patrik/Dropbox/Workspace/RMS/RMS/'
sys.path.insert(0, RMS_DIR)
from Formats import FTPdetectinfo as ftp

import SimulationTools as st

def thetaToH(theta):
	''' Converts the theta angle to height (elevation). 

		Arguments:
			theta: [float] Theta angle, measured from the zenith to the ground. 

		Returns:
			height: [float] The height corresponding to a given theta angle. 
	'''

	height = 90 - theta

	return height


def phiToAz(phi):
	''' Converts the phi angle to azimuth. 

		Arguments:
			phi: [float] Phi angle. 

		Returns:
			azim: [float] Azimuth. 
	'''

	azim = (90 - phi) % 360

	return azim


def angleDist(pos_a, pos_b):
	''' Calculates the distance between two points in a spherical coordinate system. 

		Arguments:
			pos_a: [tuple of floats] First position composed of (height, azimuth) coordinates. (deg, deg)
			pos_b: [tuple of floats] Second position composed of (height, azimuth) coordinates. (deg, deg)

		Returns:
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
    ''' Finds all *FITS files in a given directory. 

        Arguments:
            fits_dir: [string] The directory name. 

        Returns:
            fits_files: [list of strings] List containing the filenames in the directory. 
    '''

    fits_files = []

    for root, dirs, files in os.walk(fits_dir):
        for file in files:
            if file.endswith('.fits'):
                fits_files.append(file)

    return fits_files

def findTXT(txt_dir):

    txt_files = []

    for root, dirs, files in os.walk(txt_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(file)

    return txt_files


def phiList(data_dir, data_name, fits_files):
    ''' Finds all meteor angles for a given set of meteors. 

        Arguments:
            data_dir: [string] The directory of the FTPdetectinfo file.
            data_name: [string] The name of the FTPdetectinfo file.
            fits_files: [list of strings] List containing the *FITS filenames in the directory. 

        Returns:
            phi_arr: [list of floats] All of the found meteor angles.
    '''

    # Read the FTP
    data = ftp.readFTPdetectinfo(data_dir, data_name)

    phi_arr = []

    for i in range(len(data)):

        # Unpack one meteor
        meteor_data = data[i]

        # Find the angle
        ff_name, phi = meteor_data[0], meteor_data[10]

        # Check if in a given array
        if ff_name in fits_files:
            phi_arr.append(phi)

    return phi_arr


def correctDataSpatial(data_dir, data_name, fits_files, save_dir, save_name, img_y):
    ''' Corrects a given FTPdetectinfo file using the spatial correction (see SimulationTools).

        Arguments:
            data_dir: [string] The directory of the FTPdetectinfo file.
            data_name: [string] The name of the FTPdetectinfo file.
            fits_files: [list of strings] List containing the *FITS filenames in the directory.
            save_dir: [string] The directory where the new FTPdetectinfo file is saved. 
            save_name: [string] The name of the new FTPdetectinfo. 
            img_y: [float] The vertical image resolution. (px)

        Returns:
            None
    ''' 

    # Read the FTP
    data = ftp.readFTPdetectinfo(data_dir, data_name)

    meteor_list_spat = []

    for i in range(len(data)):

        # Unpack one meteor
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

    # Write the new FTPdetectinfo
    ftp.writeFTPdetectinfo(meteor_list_spat, save_dir, save_name, save_dir, cam_code, fps, calibration=None, celestial_coords_given=True)

    return None


def correctDataTemporal(data_dir, data_name, fits_files, save_dir, save_name, img_y, time_mark):
    ''' Corrects a given FTPdetectinfo file using the spatial correction (see SimulationTools).

        Arguments:
            data_dir: [string] The directory of the FTPdetectinfo file.
            data_name: [string] The name of the FTPdetectinfo file.
            fits_files: [list of strings] List containing the *FITS filenames in the directory.
            save_dir: [string] The directory where the new FTPdetectinfo file is saved. 
            save_name: [string] The name of the new FTPdetectinfo. 
            img_y: [float] The vertical image resolution. (px)
            time_mark: [string] Indicates the position of the time mark for each frame. 'start' if the time mark is
                at the start of the frame, 'end' if it is on the end of the frame.

        Returns:
            None
    ''' 

    # Read the FTP
    data = ftp.readFTPdetectinfo(data_dir, data_name)

    meteor_list_temp = []

    for i in range(len(data)):

        # Unpack one meteor
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
            t_arr_corr = st.timeCorrection(coord_arr, img_y, fps, 0, time_mark = time_mark, t_init = t_arr[0])
            frame_n_arr_corr = [x*fps for x in t_arr_corr]

            # Construct centroids array
            centroids_temp = np.c_[frame_n_arr_corr, col_arr, row_arr, ra_arr, dec_arr, azim_arr, elev_arr, inten_arr, mag_arr]

            # Construct meteor 
            meteor_temp = [ff_name, meteor_no, rho, phi, centroids_temp]

            meteor_list_temp.append(meteor_temp)

    # Extract camera code and FPS
    cam_code = data[0][1]
    fps = data[0][4]

    # Write the new FTPdetectinfo
    ftp.writeFTPdetectinfo(meteor_list_temp, save_dir, save_name, save_dir, cam_code, fps, calibration=None, celestial_coords_given=True)

    return None


def correctCelestial(rms_dir, project_dir, data_dir, save_name):
    ''' Applies the astrometric conversions (corrects the celestial coordinates) to a given data directory. 

        Arguments:
            rms_dir: [string] Local RMS directory location. 
            project_dir: [string] Local RollingShutterSim repository location.
            data_dir: [string] The directory on which the astrometric conversions are applied. 
            save_name: [string] The directory where the calibrated data is saved. 
    
        Returns:
            None
    '''

    os.chdir(rms_dir)
    os.system('python -m RMS.Astrometry.ApplyAstrometry ' + data_dir + save_name)
    os.chdir(project_dir + 'Source/')

    return None

def CAMOtoAVT(data_dir, data_name, save_dir):

    # The only used columns are time, theta and phi
    columns_used = (1, 6, 7)

    # Load the data into arrays
    t_arr, theta_arr, phi_arr = np.loadtxt(data_dir + data_name, comments='#', usecols=columns_used, unpack=True)

    # Convert (theta, phi) to (height, azimuth)
    h_arr = thetaToH(theta_arr)
    azim_arr = phiToAz(phi_arr)

    # Distance array
    dist_arr = []

    # Time difference array 
    delta_t_arr = []

    # Go through all coordinates and get angular distance; get time difference
    n_coord = len(h_arr)

    for coord_i in range(1, n_coord):

        # Create coordinate tuples
        pos_1 = (h_arr[coord_i-1], azim_arr[coord_i-1])
        pos_2 = (h_arr[coord_i], azim_arr[coord_i]) 

        # Get coordinate distance
        dist = angleDist(pos_1, pos_2)
        dist_arr.append(dist)

        # Get time difference
        delta_t = t_arr[coord_i] - t_arr[coord_i-1]
        delta_t_arr.append(delta_t)


    # Output arrays
    ang_vel_arr = []
    final_t_arr = []

    # Check if two subsequent entries in the data files have the same time
    for coord_i in range(n_coord - 1):

        if delta_t_arr[coord_i] != 0:
            ang_vel_arr.append(dist_arr[coord_i] / delta_t_arr[coord_i])
            final_t_arr.append(t_arr[coord_i+1])

    # Assume that the initial velocity is equal to the subsequent velocity
    ang_vel_arr.insert(0, ang_vel_arr[0])
    final_t_arr.insert(0, t_arr[0])

    # Save angular velocity data array
    np.savez(save_dir + data_name[:-4] + '.npz', *[final_t_arr, ang_vel_arr])

    return None


def FTPtoAVT(data_dir, data_name, fits_files, save_dir, corr_type):
    ''' Reads the data from a given FTPdetectinfo file and computes angular velocities of meteors for that file. 

        Arguments:
            data_dir: [string] The directory of the FTPdetectinfo file.
            data_name: [string] The name of the FTPdetectinfo file.
            fits_files: [list of strings] List containing the *FITS filenames in the directory.

        Returns:
            fint_arr, av_arr: [two arrays of floats] The time coordinates and angular velocities of meteors.
    '''

    # Read the FTP
    data = ftp.readFTPdetectinfo(data_dir, data_name)

    for i in range(len(data)):

        # Unpack one meteor
        meteor_data = data[i]

        # Only need its name, measurements and FPS
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

            # Construct AV array and final time array
            for coord_i in range(n_coord-1):

                av = dist_arr[coord_i]/deltat_arr[coord_i]
                av_arr.append(av)

                fint_arr.append(t_arr[coord_i+1])

            # Initial time coordinate and velocity
            fint_arr.insert(0, t_arr[0])
            av_arr.insert(0, av_arr[0])

            np.savez(save_dir + ff_name[:-5] + '_' + corr_type + '.npz', *[fint_arr, av_arr])


    return None