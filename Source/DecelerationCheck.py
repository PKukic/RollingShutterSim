''' Check if the coordinateCorrection function works OK for decelerating meteors 
'''

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import SimulationTools as st 
import Parameters as par
# import numpy as np
		
# Used for testing
show_plots = False
rolling_shutter = True
noise = 0

# Initial parameters of the meteor
omega = 10
phi = 50
t_meteor = 0.5

# Deceleration parameters of the meteor
a = 1
v_start = omega
v_finish = omega/5

# Form deceleration parameters array
dec_arr = [a, st.getparam(a, v_start, v_finish, t_meteor)]
print(dec_arr)

# Check the meteor's initial parameters
print('Meteor velocity: {:.2f}'.format(omega))
print('Meteor angle: {}'.format(phi))

# Get centroid coordinates from rolling shutter simulation
print('Simulating rolling shutter meteor...')
time_rolling_coordinates, centroid_rolling_coordinates, model_rolling_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
	omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise, par.offset, dec_arr, show_plots)

# Check if the meteor is outside of the image 
if (time_rolling_coordinates, centroid_rolling_coordinates, model_rolling_coordinates) != (-1, -1, -1):

	# Correct the rolling shutter centroid coordinates
	print('Correcting centroid coordinates...')
	centroid_rolling_coordinates = st.coordinateCorrection(time_rolling_coordinates, centroid_rolling_coordinates, \
		par.img_y, par.fps, version = 'v_corr')


	print('Calculating average difference...')
	# Calculate average difference between the centroid global and rolling shutter coordinates
	diff_avg = st.centroidAverageDifference(model_rolling_coordinates, centroid_rolling_coordinates)

	print('Average difference between centroid global and centroid rolling shutter points: {:.2f} [px]'.format(diff_avg))

	print('Average difference: {:.2f}'.format(diff_avg))	