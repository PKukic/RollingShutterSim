''' Check if the coordinateCorrection function works OK for decelerating meteors 
'''

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import SimulationTools as st 
import Parameters as par
import matplotlib.pyplot as plt 
# import numpy as np
		
# Used for testing
show_plots = False
rolling_shutter = True
noise = 0
time_mark = 'beginning'

# Initial parameters of the meteor
omega = 50
phi = 45
t_meteor = 0.5

# Deceleration parameters of the meteor
a = 1
v_start = omega
v_finish = omega*0.9


# Form deceleration parameters array
dec_arr = [a, st.getparam(a, v_start, v_finish, t_meteor)]
print(dec_arr)

# Check the meteor's initial parameters
print('Meteor velocity: {:.2f}'.format(omega))
print('Meteor angle: {}'.format(phi))

# Get centroid coordinates from rolling shutter simulation
print('Simulating rolling  shutter meteor...')
time_rolling_coordinates, centroid_rolling_coordinates, model_rolling_coordinates_spat = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
	omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise, par.offset, dec_arr, show_plots)

# Correct the rolling shutter centroid coordinates
print('Correcting centroid coordinates...')
centroid_rolling_coordinates_v = st.coordinateCorrection(time_rolling_coordinates, centroid_rolling_coordinates, \
	par.img_y, par.fps, version = 'v')


centroid_rolling_coordinates_vcorr = st.coordinateCorrection(time_rolling_coordinates, centroid_rolling_coordinates, \
	par.img_y, par.fps, version = 'v_corr')

show_plots = False

diff_arr = []
diff_v_arr = []
diff_vcorr_arr = []

for i in range(len(centroid_rolling_coordinates)):
	print(model_rolling_coordinates_spat[i], centroid_rolling_coordinates[i], centroid_rolling_coordinates_vcorr[i], centroid_rolling_coordinates_v[i])


	diff_arr.append(st.centroidDifference(model_rolling_coordinates_spat[i], centroid_rolling_coordinates[i]))
	diff_v_arr.append(st.centroidDifference(model_rolling_coordinates_spat[i], centroid_rolling_coordinates_v[i]))
	diff_vcorr_arr.append(st.centroidDifference(model_rolling_coordinates_spat[i], centroid_rolling_coordinates_vcorr[i]))

	plt.scatter(model_rolling_coordinates_spat[i][0], model_rolling_coordinates_spat[i][1], label = 'model')
	plt.scatter(centroid_rolling_coordinates[i][0], centroid_rolling_coordinates[i][1], label = 'centroid')
	plt.scatter(centroid_rolling_coordinates_vcorr[i][0], centroid_rolling_coordinates_vcorr[i][1], label = 'corrected vel')
	plt.scatter(centroid_rolling_coordinates_v[i][0], centroid_rolling_coordinates_v[i][1], label = 'uncorrected vel')

	plt.legend(loc = 'best')

	plt.xlim((0, par.img_x))
	plt.ylim((0, par.img_y))

	plt.savefig('temp/{}.png'.format(i))

	# plt.show()
	
# time_rolling_coordinates, centroid_rolling_coordinates, model_rolling_coordinates_spat = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
	# omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise, par.offset, dec_arr, show_plots, centroid_rolling_coordinates_vcorr)



# Correct the rolling shutter temporal coordinates
print('Correcting temporal coordinates...')
time_rolling_coordinates_corr = st.timeCorrection(centroid_rolling_coordinates, par.img_y, par.fps, t_meteor, time_mark)

# Calculate the model points' coordinates
print('Getting model coordinates...')
model_rolling_coordinates_temp = []

for i in range(len(time_rolling_coordinates)):

	x_model, y_model = st.drawPoints(time_rolling_coordinates_corr[i], par.img_x/2, par.img_y/2, par.scale, phi, omega, dec_arr, t_meteor)
	model_rolling_coordinates_temp.append((x_model, y_model))

vel_v = st.getVelocity(time_rolling_coordinates, centroid_rolling_coordinates_v)
vel_vcorr = st.getVelocity(time_rolling_coordinates, centroid_rolling_coordinates_vcorr)
vel_model = st.getVelocity(time_rolling_coordinates, model_rolling_coordinates_spat)

plt.plot(time_rolling_coordinates, vel_v, label = 'uncorrected velocity')
plt.plot(time_rolling_coordinates, vel_vcorr, label = 'corrected velocity')
plt.plot(time_rolling_coordinates, vel_model, label = 'model')

plt.xlabel('Time [s]')
plt.ylabel('Velocity [px/s]')

plt.legend(loc = 'best')

plt.savefig('temp/vel_time.png')

plt.show()


# diff_v = [st.centroidDifference(x, y) for x, y in zip(model_rolling_coordinates_spat, centroid_rolling_coordinates_v)]
# diff_vcorr = [st.centroidDifference(x, y) for x, y in zip(model_rolling_coordinates_spat, centroid_rolling_coordinates_vcorr)]

plt.plot(time_rolling_coordinates, diff_arr, label = 'centroid')
plt.plot(time_rolling_coordinates, diff_v_arr, label = 'uncorrected velocity')
plt.plot(time_rolling_coordinates, diff_vcorr_arr, label = 'corrected velocity')

plt.xlabel('Time [s]')
plt.ylabel('Centroid offset [px]')

plt.legend(loc = 'best')

plt.savefig('temp/offset_time.png')

plt.show()


# print('Calculating average difference...')
# diff_avg_spat = st.centroidAverageDifference(centroid_rolling_coordinates_corr, model_rolling_coordinates_spat)

# print('Average difference (spatial correction): {:.2f} [px]'.format(diff_avg_spat))

# print('Calculating average difference...')
# diff_avg_temp = st.centroidAverageDifference(model_rolling_coordinates_temp, centroid_rolling_coordinates)

# print('Average difference (temporal correction): {:.2f} [px]'.format(diff_avg_temp))


