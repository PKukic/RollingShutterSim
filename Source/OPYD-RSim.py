""" Simulation of a meteor captured by a rolling shutter camera.
"""


# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import numpy as np

import SimulationTools as st
import Parameters as par

# Check time -- done!
model_coordinates, centroid_coordinates = st.pointsCentroidAndModel(*par.param)
# print(st.centroidAverageDifference(model_coordinates, centroid_coordinates))

rolling_shutter = True

### Difference as a function of multiple parameters ###

# Counter
num_omega = 0
for omega_iter in par.omega_arr:

    # Final data array
    phi_ycentr_diff_array = []

    for phi_iter in par.phi_array:

        # LIST of centroid and model coordinates
        centroid_coordinates, model_coordinates = st.pointsCentroidAndModel(rolling_shutter, par.t_meteor, phi_iter, \
            omega_iter, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, par.noise_scale, par.offset, par.show_plots)
    

        # Size of frame number array
        frame_num_range = len(centroid_coordinates)

        print("Number of frames: {};".format(frame_num_range))

        # Generating and appending model-centroid points difference
        for frame_num in range(frame_num_range):

            # Parameters
            diff = st.centroidDifference(centroid_coordinates[frame_num], model_coordinates[frame_num])
            if(diff > 250):
                print("DIFFERENCE GREATER THAN 250 PX")

            y_centr = centroid_coordinates[frame_num][1]

            x_model = centroid_coordinates[frame_num][0]
            y_model = centroid_coordinates[frame_num][1]

            if x_model >= 0 and y_model >= 0 and x_model <= par.img_x and y_model <= par.img_y:
                phi_ycentr_diff_array.append((phi_iter, y_centr, diff))
                # Checking parameters
                print("Velocity: {:.2f} Angle: {:.2f}; Y coordinate: {:.2f}; Difference: {:.2f};".format(omega_iter, phi_iter, y_centr, diff))
                
                # Checking coordinates
                print("\tCentroid coordinates: ({:.2f}, {:.2f})".format(centroid_coordinates[frame_num][0], \
                    centroid_coordinates[frame_num][1]))

                print("\tModel coordinates: ({:.2f}, {:.2f})".format(model_coordinates[frame_num][0], \
                    model_coordinates[frame_num][1]))


    # Variable arrays
    phi_data = [point[0] for point in phi_ycentr_diff_array]
    ycentr_data = [point[1] for point in phi_ycentr_diff_array]
    diff_data = [point[2] for point in phi_ycentr_diff_array]

    # Checking
    print(len(phi_data), len(ycentr_data), len(diff_data))

    # Saving data
    np.savez('../Data/Tests/OPYD-R/data_opyd_rolling{}.npz'.format(num_omega), *[par.omega_pxs, phi_data, ycentr_data, diff_data])

    num_omega += 1
