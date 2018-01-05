import SimulationTools as st 
import Parameters as par
import numpy as np

# Customized parameters
show_plots = False

deltav_arr = []

for phi in par.phi_array:

	print("Current angle: {}".format(phi))

	print("Getting time from angle...")
	t_meteor = st.timeFromAngle(phi, par.omega, par.img_x, par.img_y, par.scale, par.fps)

	print("Simulating meteor...")
	time_coordinates, centroid_coordinates, model_coordinates = st.pointsCentroidAndModel(par.rolling_shutter, t_meteor, phi, \
    	par.omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, par.noise_scale, par.offset, show_plots)

	print("Correcting coordinates...")
	centroid_coordinates, omega_pxs = st.coordinateCorrection(t_meteor, centroid_coordinates, par.img_y, par.fps)

	print("Done!")

	# Calculate velocity error
	delta_v = omega_pxs - par.omega * par.scale

	deltav_arr.append(delta_v)

# Save data
np.savez('../Data/data_velocity_error.npz', *[par.phi_array, deltav_arr])