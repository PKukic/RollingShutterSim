''' Simulate the difference of the model and corrected centroid points with respect to meteor
	angle and angular velocity, for each noise scale level.
'''

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import numpy as np
import SimulationTools as st 
import Parameters as par 


# Fixed meteor velocity, used only for testing
omega = 40

# Fixed meteor angle, used for testing
phi = 200

# Used for testing
show_plots = False


# Get duration of meteor (the meteor is crossing the entire image)
print('Getting time from angle...')
t_meteor = st.timeFromAngle(phi, omega, par.img_x, par.img_y, par.scale, par.fps)

# Get time and centroid coordinates from global shutter
print('Simulating global shutter meteor...')
rolling_shutter = False
time_global_coordinates, centroid_global_coordinates, model_global_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
	omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, par.noise_scale, par.offset, show_plots)

# Get time and centroid coordinates from rolling shutter
print('Simulating rolling shutter meteor...')
rolling_shutter = True
time_rolling_coordinates, centroid_rolling_coordinates, model_rolling_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
	omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, par.noise_scale, par.offset, show_plots)

# Checking length of global and rolling shutter arrays BEFORE the correction
print(len(model_rolling_coordinates), len(model_global_coordinates))

# Check if meteor is outside of the image
if (time_rolling_coordinates, centroid_rolling_coordinates, model_rolling_coordinates) != (-1, -1, -1):

	print('Correcting centroid coordinates...')
	# Correct rolling shutter centroid coordinates
	centroid_rolling_coordinates = st.coordinateCorrection(time_rolling_coordinates, centroid_rolling_coordinates, \
		phi, par.img_y, par.fps)

	print('Calculating average difference...')

	# Checking length of global and rolling shutter arrays AFTER the correction
	print(len(centroid_rolling_coordinates), len(centroid_global_coordinates))

	# Calculate average difference between centroid global and rolling shutter coordinates
	# diff_avg = st.centroidAverageDifference(centroid_global_coordinates, centroid_rolling_coordinates)

	# print('Average difference: {:.2f}'.format(diff_avg))

else:
	print("Meteor is outside of the image!")