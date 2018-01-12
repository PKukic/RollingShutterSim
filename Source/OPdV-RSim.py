''' Simulate the velocity error depending on the meteor velocity and meteor angle on the image,
	in case of using the rolling shutter camera.
'''

import SimulationTools as st 
import Parameters as par
import numpy as np

# Parameters used only for this simulation
show_plots = False

# Final data array
omega_phi_deltav_arr = []

# Go through all meteor velocities and angles
for omega in par.omega_odn_arr:

	for phi in par.phi_array:

		# Get the time that takes the meteor to cross the entire image
		print("Getting time from angle...")
		t_meteor = st.timeFromAngle(phi, omega, par.img_x, par.img_y, par.scale, par.fps)


		# Get the model and centroid coordinates
		print("Simulating meteor...")
		rolling_shutter = True
		time_coordinates, centroid_coordinates, model_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
	    	omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, par.noise_scale, par.offset, show_plots)

		print('Simulation done!')

		# Check if meteor is outside of the image
		if (time_coordinates, centroid_coordinates, model_coordinates) != (-1, -1, -1):


			# Correct the centroid coordinates, assuming the meteor velocity obtained from the centroid coordinates
			# is constant
			print("Correcting coordinates...")
			centroid_coordinates_corr, omega_pxs = st.coordinateCorrection(t_meteor, centroid_coordinates, par.img_y, par.fps)

			print('Generating data...')
			
			n = len(time_coordinates)

			# Calculate meteor velocity from the centroid coordinates, using the rolling shutter camera
			total_time_rolling = time_coordinates[n - 1] - time_coordinates[0]
			total_r_rolling = st.centroidDifference(centroid_coordinates[n - 1], centroid_coordinates[0])
			avg_v_rolling = total_r_rolling / total_time_rolling

			# Calculate the actual velocity
			true_v_global = omega * par.scale
			
			# Calculate velocity error
			delta_v = true_v_global - avg_v_rolling

			print("Velocity: {:.2f}; angle: {}; velocity error: {:.2f}".format(true_v_global, phi, delta_v))

			omega_phi_deltav_arr.append((avg_v_rolling, phi, delta_v))
			

# Split data into arrays
omega_data = [point[0] for point in omega_phi_deltav_arr]
phi_data = [point[1] for point in omega_phi_deltav_arr]
deltav_data = [point[2] for point in omega_phi_deltav_arr]

# Save data
np.savez('../Data/OPdV-R/data_velocity_error3D.npz', *[omega_data, phi_data, deltav_data])