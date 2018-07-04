''' Check if the coordinateCorrection function works OK for decelerating meteors 
'''

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import SimulationTools as st 
import Parameters as par
import matplotlib.pyplot as plt 
import numpy as np
		
# Used for testing
show_plots = False
rolling_shutter = True
noise = 0

# Initial parameters of the meteor
omega = 50
phi = 45

# Deceleration parameters of the meteor
a = 1
v_start = omega
v_finish = omega*0.9

t_arr = np.arange(5/par.fps, 15/par.fps, 5/par.fps/30)

diff_avg_arr = []

for i in range(len(t_arr)):

	t_meteor = t_arr[i]

	print('{:.4f}'.format(t_meteor))

	# Form deceleration parameters array
	dec_arr = [a, st.getparam(a, v_start, v_finish, t_meteor)]
	print(dec_arr)

	# Get centroid coordinates from rolling shutter simulation
	print('Simulating rolling  shutter meteor...')
	time_rolling_coordinates, centroid_rolling_coordinates, model_rolling_coordinates_spat = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
		omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise, par.offset, dec_arr, show_plots)
	

	centroid_rolling_coordinates_vcorr = st.coordinateCorrection(time_rolling_coordinates, centroid_rolling_coordinates, \
		par.img_y, par.fps, version = 'v_corr')

	diff_vcorr_arr = []

	for i in range(len(centroid_rolling_coordinates_vcorr)):

		diff_vcorr_arr.append(st.centroidDifference(model_rolling_coordinates_spat[i], centroid_rolling_coordinates_vcorr[i]))


	diff_avg = np.average(diff_vcorr_arr)

	diff_avg_arr.append(diff_avg)

plt.plot(t_arr, diff_avg_arr, 'bo-')

plt.xlabel('Meteor duration [s]')
plt.ylabel('Centroid offset [px]')

plt.show()