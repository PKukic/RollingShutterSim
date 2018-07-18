''' Check if the coordinateCorrection function works OK for decelerating meteors [ONLY DECLERATING METEORS TESTED]
'''

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import SimulationTools as st 
import Parameters as par
import matplotlib.pyplot as plt

plt.ioff() 
		
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
a = 1e-3
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

# Correct the rolling shutter centroid coordinates (uncorrected velocity)
print('Correcting centroid coordinates...')
centroid_rolling_coordinates_v = st.coordinateCorrection(time_rolling_coordinates, centroid_rolling_coordinates, \
	par.img_y, par.fps, version = 'v')

# Apply the spatial correction while also correcting the centroid velocity
centroid_rolling_coordinates_vcorr = st.coordinateCorrection(time_rolling_coordinates, centroid_rolling_coordinates, \
	par.img_y, par.fps, version = 'v_corr')

# Apply the temporal correction to centroid coordinates
time_rolling_coordinates_corr = st.timeCorrection(centroid_rolling_coordinates, par.img_y, par.fps, t_meteor, time_mark)

# Compute the model coordinates for the temporally corrected centroid coordinates
model_rolling_coordinates_temp = st.getModelfromTime(time_rolling_coordinates_corr, par.img_x, par.img_y, par.scale, phi, omega, dec_arr, t_meteor)


# Plot the model coordinates, and spatially corrected coordinates with corrected and uncorrected centroid velocities
for i in range(len(centroid_rolling_coordinates)):
	
	print(model_rolling_coordinates_spat[i], centroid_rolling_coordinates[i], centroid_rolling_coordinates_vcorr[i], centroid_rolling_coordinates_v[i])

	plt.scatter(model_rolling_coordinates_spat[i][0], model_rolling_coordinates_spat[i][1], label = 'model')
	plt.scatter(centroid_rolling_coordinates[i][0], centroid_rolling_coordinates[i][1], label = 'centroid')
	plt.scatter(centroid_rolling_coordinates_vcorr[i][0], centroid_rolling_coordinates_vcorr[i][1], label = ' spatial [vcorr]')
	plt.scatter(centroid_rolling_coordinates_v[i][0], centroid_rolling_coordinates_v[i][1], label = 'spatial [v]')

	plt.legend(loc = 'best')

	plt.xlim((0, par.img_x))
	plt.ylim((0, par.img_y))

	plt.savefig('../Graphs/DecelerationCheck/{}.png'.format(i))

	plt.close()

# Compute the difference between the true (global shutter) centroid positions and positions created by applying corrections to centroids

diff_arr = []
diff_v_arr = []
diff_vcorr_arr = []
diff_temp_arr = []

for i in range(len(centroid_rolling_coordinates)):

	diff_arr.append(st.centroidDifference(model_rolling_coordinates_spat[i], centroid_rolling_coordinates[i]))
	diff_v_arr.append(st.centroidDifference(model_rolling_coordinates_spat[i], centroid_rolling_coordinates_v[i]))
	diff_vcorr_arr.append(st.centroidDifference(model_rolling_coordinates_spat[i], centroid_rolling_coordinates_vcorr[i]))
	diff_temp_arr.append(st.centroidDifference(model_rolling_coordinates_spat[i], model_rolling_coordinates_temp[i]))

# Compute the velocity arrays of different centroid coordinate arrays
vel_v = st.getVelocity(time_rolling_coordinates, centroid_rolling_coordinates_v)
vel_vcorr = st.getVelocity(time_rolling_coordinates, centroid_rolling_coordinates_vcorr)
vel_temp = st.getVelocity(time_rolling_coordinates_corr, model_rolling_coordinates_temp)
vel_model = st.getVelocity(time_rolling_coordinates, model_rolling_coordinates_spat)


# Plot the velocities over time
plt.plot(time_rolling_coordinates, vel_model, label = 'model')
plt.plot(time_rolling_coordinates_corr, vel_temp, label = 'temporal')
plt.plot(time_rolling_coordinates, vel_v, label = 'spatial [v]')
plt.plot(time_rolling_coordinates, vel_vcorr, label = 'spatial [vcorr]')

plt.xlabel('Time [s]')
plt.ylabel('Velocity [px/s]')

plt.legend(loc = 'best')

plt.savefig('../Graphs/DecelerationCheck/vel_time.png')

plt.close()

# Plot the centroid offsets over time
plt.plot(time_rolling_coordinates, diff_arr, label = 'uncorrected')
plt.plot(time_rolling_coordinates_corr, diff_temp_arr, label = 'temporal')
plt.plot(time_rolling_coordinates, diff_v_arr, label = 'spatial [v]')
plt.plot(time_rolling_coordinates, diff_vcorr_arr, label = 'spatial [vcorr]')

plt.xlabel('Time [s]')
plt.ylabel('Centroid offset [px]')

plt.legend(loc = 'best')

plt.savefig('../Graphs/DecelerationCheck/offset_time.png')

plt.close()