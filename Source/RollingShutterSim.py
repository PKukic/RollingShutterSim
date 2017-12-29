""" Simulation of a meteor captured by a rolling shutter camera.
"""


# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import numpy as np

from SimulationTools import *

### Defining function parameters ###

# Using rolling shutter
rolling_shutter = False

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
scale = img_x/42

#  Number of frames per second
fps = 25

# Meteor's angular velocity (deg/s)
omega = 50

# Angular velocity array in px/s #(logarithmic)
omega_pxs = np.logspace(np.log10(1), np.log10(1500), 10)
# Angular velocity array in deg/s #(logarithmic)
omega_arr = omega_pxs / scale

# Checking
print(omega_arr)

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
# model_coordinates, centroid_coordinates = pointsCentroidAndModel(*param)
# print(centroidAverageDifference(model_coordinates, centroid_coordinates))


### Difference as a function of multiple parameters ###

# Counter
num_omega = 0
for omega_iter in omega_arr:

    # Final data array
    phi_ycentr_diff_array = []
    # amplitude = 255/img_y*(2*omega_iter/scale)
    # print("Amplitude: {}".format(amplitude))

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
            if(diff > 250):
                print("DIFFERENCE GREATER THAN 250 PX")

            y_centr = centroid_coordinates[frame_num][1]
            
            # Checking parameters
            print("Velocity: {:.2f} Angle: {:.2f}; Y coordinate: {:.2f}; Difference: {:.2f};".format(omega_iter, phi_iter, y_centr, diff))
            
            # Checking coordinates
            print("\tCentroid coordinates: ({:.2f}, {:.2f})".format(centroid_coordinates[frame_num][0], \
                centroid_coordinates[frame_num][1]))

            print("\tModel coordinates: ({:.2f}, {:.2f})".format(model_coordinates[frame_num][0], \
                model_coordinates[frame_num][1]))

            phi_ycentr_diff_array.append((phi_iter, y_centr, diff))


    # Variable arrays
    phi_data = [point[0] for point in phi_ycentr_diff_array]
    ycentr_data = [point[1] for point in phi_ycentr_diff_array]
    diff_data = [point[2] for point in phi_ycentr_diff_array]

    # Checking
    print(len(phi_data), len(ycentr_data), len(diff_data))

    # Saving data
    np.savez('../Data/Tests/OPYD-R/data_opyd_rolling{}.npz'.format(num_omega), *[omega_pxs, phi_data, ycentr_data, diff_data])

    num_omega += 1
