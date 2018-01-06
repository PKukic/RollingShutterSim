import SimulationTools as st 
import Parameters as par
import numpy as np
import matplotlib.pyplot as plt

# Customized parameters
show_plots = False

# deltav_arr = []

phi_array = np.arange(0, 361, 10)

for phi in phi_array:

	print("Current angle: {}".format(phi))

	print("Getting time from angle...")
	t_meteor = st.timeFromAngle(phi, par.omega, par.img_x, par.img_y, par.scale, par.fps)

	print("Simulating meteor...")
	rolling_shutter = True
	time_coordinates, centroid_coordinates, model_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
    	par.omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, par.noise_scale, par.offset, show_plots)

	rolling_shutter = False
	time_global_coordinates, centroid_global_coordinates, model_global_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
    	par.omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, par.noise_scale, par.offset, show_plots)

	print("Correcting coordinates...")
	centroid_coordinates_corr, omega_pxs = st.coordinateCorrection(t_meteor, centroid_coordinates, par.img_y, par.fps)

	print('Generating data...')

	delta_t = []
	delta_r = []
	delta_rc = []
	delta_v = []
	delta_vc = []

	delta_t.append(0)
	delta_r.append(0)
	delta_rc.append(0)
	delta_v.append(0)
	delta_vc.append(0)
	
	cnt_t = 0
	cnt_r = 0
	cnt_rc = 0

	for i in range(len(time_coordinates) - 1):
		dt = time_coordinates[i + 1] - time_coordinates[i]
		dr = st.centroidDifference(centroid_global_coordinates[i + 1], centroid_global_coordinates[i])
		d_rc = st.centroidDifference(centroid_coordinates_corr[i + 1], centroid_coordinates_corr[i])

		delta_t.append(dt + cnt_t)
		delta_r.append(dr + cnt_r)
		delta_rc.append(d_rc + cnt_rc)
		
		cnt_t += dt
		cnt_r += dr
		cnt_rc += d_rc

		delta_v.append(delta_r[i + 1] / delta_t[i + 1])
		delta_vc.append(delta_rc[i + 1] / delta_t[i + 1])

	print("Generating D/T graph...")

	# plt.plot(delta_t, delta_r, 'ro--', label = 'regular coordinates')
	# plt.plot(delta_t, delta_rc, 'bo--', label = 'corrected coordinates')

	# plt.xlabel('Time [s]')
	# plt.ylabel('Distance [px]')
	# plt.title('Meteor angle $\phi$: {} [deg]'.format(phi))
	# plt.legend(loc = 'lower right')
	# plt.axis('tight')

	# plt.savefig('../Graphs/Velocity error/Distance-time plots/plot_distance_time_{}'.format(phi))

	# plt.show()

	print('Generating V/T graph...')
	
	plt.plot(delta_t, delta_v, 'ro--', label = 'regular velocity')
	plt.plot(delta_t, delta_vc, 'bo--', label = 'corrected velocity')

	plt.xlabel('Time [s]')
	plt.ylabel('Velocity [px/s]')
	plt.title('Meteor angle $\phi$: {} [deg]'.format(phi))
	plt.legend(loc = 'lower right')
	plt.axis('tight')

	plt.savefig('../Graphs/Velocity error/Velocity-time plots/plot_velocity_time_{}'.format(phi))

	plt.show()


# Save data
# np.savez('../Data/data_velocity_error.npz', *[par.phi_array, deltav_arr])