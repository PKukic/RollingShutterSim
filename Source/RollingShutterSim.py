""" Simulation of a meteor captured by a rolling shutter camera.
"""


# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import numpy as np

from SimulationTools import *


if __name__ == "__main__":

    ### Defining function parameters ###

    # Using rolling shutter
    rolling_shutter = True

    # Meteor duration
    t_meteor = 0.5

    # Meteor angle counterclockwise from the Y axis (deg)
    phi = 120

    # Meteor angle array
    phi_array = np.arange(0, 361)

    # Meteor's angular velocity (deg/s)
    omega = 50

    # Angular velocity array
    omega_arr = np.arange(1, 50.5, 0.5)

    # Image size
    img_x = 1280
    img_y = 720

    # Pixel scale in px/deg
    scale = img_x/64

    #  Number of frames per second
    fps = 25

    # Standard deviation along X and Y axis
    sigma_x = 2
    sigma_y = 2

    # Scale of background noise
    noise_scale = 10

    # Scale of background noise array
    noise_scale_arr = [0, 5, 10, 20]

    # Level of background offset
    offset = 20

    # Plot individual frames?
    show_plots = False


    # List of all unified parameters
    param = [rolling_shutter, t_meteor, phi, omega, img_x, img_y, scale, fps, sigma_x, sigma_y, noise_scale, offset, show_plots]


    # Check time -- done!
    # pointsCentroidAndModel(*param)


    ### Difference as a function of frame number and meteor angle on image ###
    
    # Final data array
    phi_num_diff_array = []


    for phi_iter in phi_array:

        # LIST of centroid and model coordinates
        centroid_coordinates, model_coordinates = pointsCentroidAndModel(rolling_shutter, t_meteor, phi_iter, omega, img_x, img_y, scale, fps, sigma_x, sigma_y, noise_scale, offset, show_plots)
        
        # Size of frame number array
        frame_num_range = len(centroid_coordinates)

        # Generating and appending model-centroid points difference
        for frame_num in range(frame_num_range):
            diff = centroidDifference(centroid_coordinates[frame_num], model_coordinates[frame_num])
            phi_num_diff_array.append((phi_iter, frame_num, diff))

            # Checking parameters
            print("Meteor angle: {} frame number: {} difference: {:.2f}".format(phi_iter, frame_num, diff))


    # Variable arrays
    phi_data = [point[0] for point in phi_num_diff_array]
    frame_num_data = [point[1] for point in phi_num_diff_array]
    diff_data = [point[2] for point in phi_num_diff_array]

    # Generate frame number array
    frame_num_size = len(frame_num_data)/phi_array.size
    frame_array = np.arange(frame_num_size)
    
    # Generate frame number/angle meshgrid
    pp, ff = np.meshgrid(phi_array, frame_array)

    # Reshape the difference array so that it is compatible with the meshgrid
    diff_data = np.reshape(diff_data, (phi_array.size, frame_num_size))

    
    # Checking sizes
    print("Size of frame num array: {}".format(frame_array.size))
    print("Size of phi array: {}".format(phi_array.size))
    print("Shape of difference data: {}".format(diff_data.shape))


    # Saving data
    np.savez('data_phi_frame_diff_rolling.npz', *[pp, ff, diff_data])