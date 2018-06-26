""" Plot data obtained by the omega-difference-noise simulation for the rolling shutter camera, with the centroid coordinates corrected.
"""
# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import matplotlib.pyplot as plt
import numpy as np
import Parameters as par

# Load data
data = np.load('../Data/ODN/data_odn_v_rolling.npz')

# Unpack and set array names
omega_arr = data['arr_0'] * par.fps
noise0_data = data['arr_1']
noise1_data = data['arr_2']
noise2_data = data['arr_3']
noise3_data = data['arr_4']

# Plot all four noise arrays
plt.plot(omega_arr, noise0_data, 'c-', label = '$\sigma$ = {}'.format(par.noise_scale_arr[0]))
plt.plot(omega_arr, noise1_data, 'r-', label = '$\sigma$ = {}'.format(par.noise_scale_arr[1]))
plt.plot(omega_arr, noise2_data, 'g-', label = '$\sigma$ = {}'.format(par.noise_scale_arr[2]))
plt.plot(omega_arr, noise3_data, 'b-', label = '$\sigma$ = {}'.format(par.noise_scale_arr[3]))

# Label the plot, set plot title, set legend
plt.legend(loc = 'lower right')
plt.xlabel("Angular velocity [px/s]")
plt.ylabel("Average model-centroid point difference [px]")
plt.title("Meteor angle {} [deg] (uncorrected velocity)".format(par.phi))


# Configure the plot axis
plt.axis('tight')

# Save and show plot
plt.savefig('../Graphs/ODN/graph_odn_v_rolling.png')
plt.show()