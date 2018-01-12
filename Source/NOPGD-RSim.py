''' Simulate the difference of the model and corrected centroid points with respect to meteor
	angle and angular velocity, for each noise scale level.
'''

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import SimulationTools as st 
import Parameters as par 


# Fixed meteor velocity, used only for testing
omega = 40

# Fixed meteor angle, used for testing
phi = 315

# Used for testing
show_plots = False

print('Meteor velocity: {:.2f}'.format(omega * par.scale))

print('Meteor angle: {}'.format(phi))

# Get duration of meteor (the meteor is crossing the entire image)
print('Getting time from angle...')
t_meteor = st.timeFromAngle(phi, omega, par.img_x, par.img_y, par.scale, par.fps)

# Get time and centroid coordinates from global shutter
'''
print('Simulating global shutter meteor...')
rolling_shutter = False
time_global_coordinates, centroid_global_coordinates, model_global_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
	omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, par.noise_scale, par.offset, show_plots)
'''

# Get time and centroid coordinates from rolling shutter
print('Simulating rolling shutter meteor...')
show_plots = True
rolling_shutter = True
time_rolling_coordinates, centroid_rolling_coordinates, model_rolling_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
	omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, par.noise_scale, par.offset, show_plots)

# Delete first two frames of global shutter meteor simulation - first frame in the rolling shutter simulation
# is skipped, and all arrays must have the same size
'''
del time_global_coordinates[0]
del centroid_global_coordinates[0]
del model_global_coordinates[0]

del time_global_coordinates[1]
del centroid_global_coordinates[1]
del model_global_coordinates[1]
'''
print('Correction factor for each centroid point (estimated):')
for i in range(len(centroid_rolling_coordinates)):
	print(st.centroidDifference(model_rolling_coordinates[i], centroid_rolling_coordinates[i]))

# Check if meteor is outside of the image
if (time_rolling_coordinates, centroid_rolling_coordinates, model_rolling_coordinates) != (-1, -1, -1):

	print('Correcting centroid coordinates...')
	# Correct rolling shutter centroid coordinates
	centroid_rolling_coordinates = st.coordinateCorrection(time_rolling_coordinates, centroid_rolling_coordinates, \
		par.img_y, par.fps)

	# print(len(centroid_global_coordinates), len(centroid_rolling_coordinates))

	# print('Calculating average difference...')

	# Calculate average difference between centroid global and rolling shutter coordinates
	# diff_avg = st.centroidAverageDifference(centroid_global_coordinates, centroid_rolling_coordinates)

	# print('Average difference: {:.2f} [px]'.format(diff_avg))
	
	print('Difference between model and centroid rolling shutter points for each frame: ')
	for i in range(len(centroid_rolling_coordinates)):
		print(st.centroidDifference(centroid_rolling_coordinates[i], model_rolling_coordinates[i]))

else:
	print("Meteor is outside of the image!")

