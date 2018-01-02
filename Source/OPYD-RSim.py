""" Simulation of a meteor captured by a rolling shutter camera.
"""


# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import numpy as np

import SimulationTools as st
import Parameters as par

# model_coordinates, centroid_coordinates = st.pointsCentroidAndModel(*par.param)
# print(st.centroidAverageDifference(model_coordinates, centroid_coordinates))

# Customised parameters
rolling_shutter = True

### Difference as a function of meteor angle, angular velocity and Y centroid coordinate ###

# Counter
num_omega = 0

for omega_iter in par.omega_arr:

    # Final data array
    phi_ycentr_diff_array = []

    for phi_iter in par.phi_array:

        # Get meteor duration (meteor is crossing the entire image)
        t_meteor = st.timeFromAngle(phi_iter, omega_iter, par.img_x, par.img_y, par.scale, par.fps)

        print("Meteor duration: {:.2f}".format(t_meteor))

        # LIST of centroid and model coordinates
        centroid_coordinates, model_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi_iter, \
            omega_iter, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, par.noise_scale, par.offset, par.show_plots)

        if centroid_coordinates != -1 and model_coordinates != -1:

            # Size of frame number array
            frame_num_range = len(centroid_coordinates)
            print("Number of frames: {};".format(frame_num_range))

            # Generating and appending model-centroid points difference
            for frame_num in range(frame_num_range):

                # Parameters
                diff = st.centroidDifference(centroid_coordinates[frame_num], model_coordinates[frame_num])

                # Model and centroid coordinates
                x_centr = centroid_coordinates[frame_num][0]
                y_centr = centroid_coordinates[frame_num][1]

                x_model = centroid_coordinates[frame_num][0]
                y_model = centroid_coordinates[frame_num][1]

                # Comparing the model coordinates before and after the check
                print("Model coordinates: ({:.2f}, {:.2f})".format(x_model, y_model))

                # Checking if the model coordinates are outside of the read image
                if x_model >= 0 and y_model >= 0 and x_model <= par.img_x and y_model <= par.img_y:
                    
                    phi_ycentr_diff_array.append((phi_iter, y_centr, diff))
                    
                    # Checking parameters
                    print("Velocity: {:.2f} Angle: {:.2f}; Y coordinate: {:.2f}; Difference: {:.2f};".format(omega_iter, phi_iter, y_centr, diff))
                    
                    # Checking coordinates
                    print("\tCentroid coordinates: ({:.2f}, {:.2f})".format(x_centr, y_centr))
                    print("\tModel coordinates: ({:.2f}, {:.2f})".format(x_model, y_model))

        else:
            print("Model coordinates are outside of the read image")


    # Variable arrays
    phi_data = [point[0] for point in phi_ycentr_diff_array]
    ycentr_data = [point[1] for point in phi_ycentr_diff_array]
    diff_data = [point[2] for point in phi_ycentr_diff_array]

    # Checking
    print(len(phi_data), len(ycentr_data), len(diff_data))

    # Saving data
    np.savez('../Data/OPYD-R/data_opyd_rolling{}.npz'.format(num_omega), *[par.omega_pxs, phi_data, ycentr_data, diff_data])

    num_omega += 1
