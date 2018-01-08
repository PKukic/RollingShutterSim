import SimulationTools as st 
import Parameters as par
import numpy as np

# Customized parameters
show_plots = False

omega_phi_deltav_arr = []

for omega in par.omega_odn_arr:

	for phi in par.phi_array:

		print("Getting time from angle...")
		t_meteor = st.timeFromAngle(phi, omega, par.img_x, par.img_y, par.scale, par.fps)

		print("Simulating meteor...")
		rolling_shutter = True
		time_coordinates, centroid_coordinates, model_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
	    	omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, par.noise_scale, par.offset, show_plots)

		print('Simulation done!')

		if (time_coordinates, centroid_coordinates, model_coordinates) != (-1, -1, -1):

			print("Correcting coordinates...")
			centroid_coordinates_corr, omega_pxs = st.coordinateCorrection(t_meteor, centroid_coordinates, par.img_y, par.fps)

			print('Generating data...')
			
			n = len(time_coordinates)

			# Calculate rolling shutter meteor velocity
			total_time_rolling = time_coordinates[n - 1] - time_coordinates[0]
			total_r_rolling = st.centroidDifference(centroid_coordinates[n - 1], centroid_coordinates[0])
			avg_v_rolling = total_r_rolling / total_time_rolling

			# Calculate actual velocity
			true_v_global = omega * par.scale
			
			# Calculate velocity error
			delta_v_true = true_v_global - avg_v_rolling

			print("Velocity: {:.2f}; angle: {}; velocity error: {:.2f}".format(true_v_global, phi, delta_v_true))

			omega_phi_deltav_arr.append((true_v_global, phi, delta_v_true))
			

# Spliting data into arrays
omega_data = [point[0] for point in omega_phi_deltav_arr]
phi_data = [point[1] for point in omega_phi_deltav_arr]
deltav_data = [point[2] for point in omega_phi_deltav_arr]

# Save data
np.savez('../Data/Velocity error/data_velocity_error3D.npz', *[omega_data, phi_data, deltav_data])