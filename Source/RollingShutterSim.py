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

    # Image size
    img_x = 1280
    img_y = 720

    # Pixel scale in px/deg
    scale = img_x/64

    #  Number of frames per second
    fps = 25

    # Meteor's angular velocity (deg/s)
    omega = 50

    # Angular velocity array
    omega_arr = np.arange(1, 50.5, scale)

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


    ### Difference as a function of multiple parameters ###
    
    num = 0

    for omega_iter in omega_arr:

        # Final data array
        phi_ycentr_diff_array = []

        for phi_iter in phi_array:

            # LIST of centroid and model coordinates
            centroid_coordinates, model_coordinates = pointsCentroidAndModel(rolling_shutter, t_meteor, phi_iter, omega_iter, img_x, img_y, scale, fps, sigma_x, sigma_y, noise_scale, offset, show_plots)
            
            # Size of frame number array
            frame_num_range = len(centroid_coordinates)

            print("Number of frames: {};".format(frame_num_range))

            # Generating and appending model-centroid points difference
            for frame_num in range(frame_num_range):

                # Parameters
                diff = centroidDifference(centroid_coordinates[frame_num], model_coordinates[frame_num])
                y_centr = centroid_coordinates[frame_num][1]
                
                # Checking parameters
                print("Velocity: {:.2f} Angle: {:.2f}; Y coordinate: {:.2f}; Difference: {:.2f};".format(omega_iter, phi_iter, y_centr, diff))

                phi_ycentr_diff_array.append((phi_iter, y_centr, diff))


        # Variable arrays
        phi_data = [point[0] for point in phi_ycentr_diff_array]
        ycentr_data = [point[1] for point in phi_ycentr_diff_array]
        diff_data = [point[2] for point in phi_ycentr_diff_array]

        # Checking
        print(len(phi_data), len(ycentr_data), len(diff_data))

        # Saving data
        np.savez('../Data/APYD-R/data_apyd_rolling{}.npz'.format(num), *[phi_data, ycentr_data, diff_data])

        num += 1