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
time_mark = 'beginning'

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
time_rolling_coordinates, centroid_rolling_coordinates, model_rolling_coordinates_spat = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
	omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise, par.offset, dec_arr, show_plots)

# Correct the rolling shutter centroid coordinates
print('Correcting centroid coordinates...')
centroid_rolling_coordinates_corr = st.coordinateCorrection(time_rolling_coordinates, centroid_rolling_coordinates, \
	par.img_y, par.fps, version = 'v_corr')

# Correct the rolling shutter temporal coordinates
print('Correcting temporal coordinates...')
time_rolling_coordinates_corr = st.timeCorrection(centroid_rolling_coordinates, par.img_y, par.fps, t_meteor, time_mark)

# Calculate the model points' coordinates
print('Getting model coordinates...')
model_rolling_coordinates_temp = []

for i in range(len(time_rolling_coordinates)):

	x_model, y_model = st.drawPoints(time_rolling_coordinates_corr[i], par.img_x/2, par.img_y/2, par.scale, phi, omega, dec_arr, t_meteor)
	model_rolling_coordinates_temp.append((x_model, y_model))



print('Calculating average difference...')
diff_avg_spat = st.centroidAverageDifference(centroid_rolling_coordinates_corr, model_rolling_coordinates_spat)

print('Average difference (spatial correction): {:.2f} [px]'.format(diff_avg_spat))

print('Calculating average difference...')
diff_avg_temp = st.centroidAverageDifference(model_rolling_coordinates_temp, centroid_rolling_coordinates)

print('Average difference (temporal correction): {:.2f} [px]'.format(diff_avg_temp))