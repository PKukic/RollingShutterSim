""" Simulate the model-centroid point difference based on the meteor velocity [omega], meteor angle [phi] and meteor centroid Y coordinate [ycentr].
"""

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import numpy as np
import SimulationTools as st
import Parameters as par


# Parameters used only for this simulation
rolling_shutter = True

# File number counter
num_omega = 0

# Go through all meteor velocities and all meteor angles
for omega in par.omega_arr:

    # Final data array
    phi_ycentr_diff_array = []

    for phi in par.phi_array:

        # Get meteor duration (meteor is crossing the entire image)
        t_meteor = st.timeFromAngle(phi, omega, par.img_x, par.img_y, par.scale, par.fps)

        print("Meteor duration: {:.2f}".format(t_meteor))

        # LIST of centroid and model coordinates for that angle
        time_coordinates, centroid_coordinates, model_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
            omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, par.noise_scale, par.offset, par.fit_param, par.show_plots)


        # Check if the meteor is outside of the image
        if (time_coordinates, centroid_coordinates, model_coordinates) != (-1, -1, -1):

            # Size of the frame array
            frame_num_range = len(centroid_coordinates)
            print("Number of frames: {};".format(frame_num_range))

            # Generate and append the model-centroid points difference
            for frame_num in range(frame_num_range):

                # Parameters
                diff = st.centroidDifference(centroid_coordinates[frame_num], model_coordinates[frame_num])

                # Model and centroid coordinates
                x_centr = centroid_coordinates[frame_num][0]
                y_centr = centroid_coordinates[frame_num][1]

                x_model = model_coordinates[frame_num][0]
                y_model = model_coordinates[frame_num][1]

                phi_ycentr_diff_array.append((phi, y_centr, diff))
                
                # Check all the parameters
                print("Velocity: {:.2f} Angle: {:.2f}; Y coordinate: {:.2f}; Difference: {:.2f};".format(omega, phi, y_centr, diff))
                
                # Check all the coordinates
                print("\tCentroid coordinates: ({:.2f}, {:.2f})".format(x_centr, y_centr))
                print("\tModel coordinates: ({:.2f}, {:.2f})".format(x_model, y_model))

        else:
            print("Model coordinates are outside of the read image")


    # Set array names
    phi_data = [point[0] for point in phi_ycentr_diff_array]
    ycentr_data = [point[1] for point in phi_ycentr_diff_array]
    diff_data = [point[2] for point in phi_ycentr_diff_array]

    # Check the length of all arrays
    print(len(phi_data), len(ycentr_data), len(diff_data))

    # Save the data as a file (a different file for a different meteor velocity)
    np.savez('../Data/OPYD-R/data_opyd_rolling{}.npz'.format(num_omega), *[par.omega_pxs, phi_data, ycentr_data, diff_data])

    num_omega += 1
