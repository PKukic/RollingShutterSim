''' Simulate the difference of the model and corrected centroid points with respect to meteor
	angle and angular velocity, for each noise scale level.
'''

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import SimulationTools as st 
import Parameters as par
import numpy as np 

# Used for testing
show_plots = False

nul_arr = [0, 0]

for noise in par.noise_scale_arr:

	omega_phi_avg_diff_arr = []

	print('Noise scale: {}'.format(noise))

	for omega in par.omega_odn_arr:

		for phi in par.phi_array:

			omega_pxs = omega * par.scale

			# Check the meteor's initial parameters
			print('Meteor velocity: {:.2f}'.format(omega_pxs))
			print('Meteor angle: {}'.format(phi))

			# Get duration of meteor (the meteor is crossing the entire image)
			print('Getting time from angle...')
			t_meteor = st.timeFromAngle	(phi, omega, par.img_x, par.img_y, par.scale, par.fps)

			# Get time and centroid coordinates from global shutter
			print('Simulating global shutter meteor...')
			rolling_shutter = False
			time_global_coordinates, centroid_global_coordinates, model_global_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
				omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise, par.offset, nul_arr, show_plots)


			# Get time and centroid coordinates from rolling shutter
			print('Simulating rolling shutter meteor...')
			show_plots = False
			rolling_shutter = True
			time_rolling_coordinates, centroid_rolling_coordinates, model_rolling_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
				omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise, par.offset, nul_arr, show_plots)

			# Delete first two frames  of global shutter meteor simulation - the first frame in the rolling shutter simulation
			# are skipped, the second is skipped while correcting the coordinates, and all arrays must have the same size
			del time_global_coordinates[:2]
			del centroid_global_coordinates[:2]
			del model_global_coordinates[:2]

			# Check if meteor is outside of the image
			if (time_rolling_coordinates, centroid_rolling_coordinates, model_rolling_coordinates) != (-1, -1, -1):

				print('Correcting centroid coordinates...')
				# Correct rolling shutter centroid coordinates
				centroid_rolling_coordinates = st.coordinateCorrection(time_rolling_coordinates, centroid_rolling_coordinates, \
					par.img_y, par.fps)

				print('Calculating average difference...')
				# Calculate average difference between the centroid global and rolling shutter coordinates
				diff_avg = st.centroidAverageDifference(centroid_global_coordinates, centroid_rolling_coordinates)

				print('Average difference between centroid global and centroid rolling shutter points: {:.2f} [px]'.format(diff_avg))
				
				# print('Difference between centroid global and centroid rolling shutter points for each frame: ')
				# for i in range(len(centroid_rolling_coordinates)):
					# print(st.centroidDifference(centroid_rolling_coordinates[i], centroid_global_coordinates[i]))

				omega_phi_avg_diff_arr.append((omega_pxs, phi, diff_avg))

			else:
				print("Meteor is outside of the image!")

	omega_data = [point[0] for point in omega_phi_avg_diff_arr]
	phi_data = [point[1] for point in omega_phi_avg_diff_arr]
	diff_avg_data = [point[2] for point in omega_phi_avg_diff_arr]

	np.savez('../Data/NOPGD-R/data_nopgd_rolling_{}.npz'.format(noise), *[omega_data, phi_data, diff_avg_data])