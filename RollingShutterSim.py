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
    phi_array = np.arange(1, 361)

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


    ### Difference as a function of frame number and meteor angle on image ###

    # Final data array
    phi_num_diff_array = []


    for phi_iter in phi_array:

        # LIST of centroid and model coordinates
        centroid_coordinates, model_coordinates = pointsCentroidAndModel(rolling_shutter, t_meteor, phi_iter, omega, img_x, img_y, scale, fps, sigma_x, sigma_y, noise_scale, offset, show_plots)
        
        # size of frame number array
        frame_num_range = len(centroid_coordinates)

        # Generating and appending model-centroid points difference
        for frame_num in range(frame_num_range):
            diff = centroidDifference(centroid_coordinates[frame_num], model_coordinates[frame_num])
            print("Meteor angle: {} frame number: {} difference: {:.2f}".format(phi_iter, frame_num, diff))
            phi_num_diff_array.append((phi_iter, frame_num, diff))


    # Variable arrays
    phi_data = [point[0] for point in phi_num_diff_array]
    frame_num_data = [point[1] for point in phi_num_diff_array]
    diff_data = [point[2] for point in phi_num_diff_array]


    print(len(phi_data), len(frame_num_data), len(diff_data))

    # Saving data
    np.savez('data_frame_angle_diff_rolling.npz', phi_data, *[frame_num_data, diff_data])


    """

    ### Average difference as a function of angular velocity ###

    # Array of average of averages
    noise_diff_arr = []

    # Number of runs
    n = 10

    for noise in noise_scale_arr:
        # Average of averages difference array
        diff_avg_avg = []
        print("Noise level: {}".format(noise))
    
        for omega_i in omega_arr:
            # Average differences array
            diff_avg = []
            for i in range(n):
                # Compute centroid and model coordinates
                centroid_coordinates, model_coordinates = pointsCentroidAndModel(rolling_shutter, t_meteor, phi, omega_i, img_x, img_y, scale, fps, sigma_x, sigma_y, noise, offset, show_plots)
                
                # Compute average distance
                diff = averageDifference(centroid_coordinates, model_coordinates)
                
                print('{} Average difference: {:.4f}'.format(i, diff))
                diff_avg.append(diff)
            print('Angular velocity[deg/s]: {:.2f} Average of difference averages: {:.4f}'.format(omega_i, np.average(diff_avg)))
            diff_avg_avg.append(np.average(diff_avg))
        noise_diff_arr.append(diff_avg_avg)
    """