''' Check the new correction method based only on correcting the time assignment to each measurement.
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
omega = 40
phi = 45
dec_arr = [0, 0]
time_mark = 'beginning'
# t_meteor = 0.37

# Check the meteor's initial parameters
print('Meteor velocity: {:.2f}'.format(omega))
print('Meteor angle: {}'.format(phi))

# print('Getting time from angle...')
t_meteor = st.timeFromAngle(phi, omega, par.img_x, par.img_y, par.scale, par.fps)

# Get centroid coordinates from rolling shutter simulation
print('Simulating rolling shutter meteor...')
time_rolling_coordinates, centroid_rolling_coordinates, model_rolling_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
	omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise, par.offset, dec_arr, show_plots)

# Testing both the spatial and temporal correction
# correction_type = 'spatial'
# frame_timestamp = 'beginning'

# coordinates = st.meteorCorrection(time_rolling_coordinates, centroid_rolling_coordinates, par.img_y, par.fps, correction_type, frame_timestamp)

# Check if the meteor is outside of the image
if (time_rolling_coordinates, centroid_rolling_coordinates, model_rolling_coordinates) != (-1, -1, -1):

	# Correct the rolling shutter time coordinates
	print('Correcting time coordinates...')
	time_rolling_coordinates = st.timeCorrection(centroid_rolling_coordinates, par.img_y, par.fps, t_meteor, time_mark)

	# Calculate the model points' coordinates
	print('Getting model coordinates...')
	model_rolling_coordinates = []

	# Image center (required for executing the drawPoints function)
	x_center = par.img_x/2
	y_center = par.img_y/2

	for i in range(len(time_rolling_coordinates)):

		x_model, y_model = st.drawPoints(time_rolling_coordinates[i], x_center, y_center, par.scale, phi, omega, dec_arr, t_meteor)
		model_rolling_coordinates.append((x_model, y_model))

		print(st.centroidDifference(model_rolling_coordinates[i], centroid_rolling_coordinates[i]))

	print('Calculating average difference between model and centroid rolling shutter points...')
	diff_avg = st.centroidAverageDifference(centroid_rolling_coordinates, model_rolling_coordinates)

	print('Average difference: {:.2f}'.format(diff_avg))