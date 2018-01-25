''' Check if the coordinateCorrection function works OK for decelerating meteors 
'''

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import SimulationTools as st 
import Parameters as par
# import numpy as np 

# Used for testing
show_plots = False
dec_arr = [1, 0.54]
rolling_shutter = True
noise = 0

# Initial parameters of the meteor
omega = 10
phi = 45

# Check the meteor's initial parameters
print('Meteor velocity: {:.2f}'.format(omega))
print('Meteor angle: {}'.format(phi))

print('Getting time from angle...')
t_meteor = st.timeFromAngle(phi, omega, par.img_x, par.img_y, par.scale, par.fps)

# Get centroid coordinates from rolling shutter simulation
print('Simulating rolling shutter meteor')
time_rolling_coordinates, centroid_rolling_coordinates, model_rolling_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
	omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise, par.offset, dec_arr, show_plots)

# Check if the meteor is outside of the image
if (time_rolling_coordinates, centroid_rolling_coordinates, model_rolling_coordinates) != (-1, -1, -1):

	# Correct the rolling shutter centroid coordinates
	print('Correcting centroid coordinates...')
	centroid_rolling_coordinates = st.coordinateCorrection(time_rolling_coordinates, centroid_rolling_coordinates, \
		par.img_y, par.fps)

	del model_rolling_coordinates[:1]

	print('Calculating average difference...')
	# Calculate average difference between the centroid global and rolling shutter coordinates
	diff_avg = st.centroidAverageDifference(model_rolling_coordinates, centroid_rolling_coordinates)

	print('Average difference between centroid global and centroid rolling shutter points: {:.2f} [px]'.format(diff_avg))

	print('Average difference: {:.2f}'.format(diff_avg))