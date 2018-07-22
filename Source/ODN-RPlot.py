""" Plot data obtained by the omega-difference-noise simulation for the rolling shutter camera
"""

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import matplotlib.pyplot as plt
import numpy as np
import Parameters as par

# Load data
data = np.load('../Data/ODN/data_odn_rolling.npz')

# Unpack and set array names
omega_arr = data['arr_0'] * par.scale
noise0_data = data['arr_1']
noise1_data = data['arr_2']
noise2_data = data['arr_3']
noise3_data = data['arr_4']

# Plot all four noise arrays
plt.plot(omega_arr, noise0_data, c = '0.8', ls = '-',  lw = 1.75, label = '$\sigma$ = {}'.format(par.noise_scale_arr[0]))
plt.plot(omega_arr, noise1_data, c = '0.8', ls = '--', lw = 1.75,  label = '$\sigma$ = {}'.format(par.noise_scale_arr[1]))
plt.plot(omega_arr, noise2_data, c = '0.5', ls = '-',  lw = 1.75, label = '$\sigma$ = {}'.format(par.noise_scale_arr[2]))
plt.plot(omega_arr, noise3_data, c = '0.5', ls = '--', lw = 1.75,  label = '$\sigma$ = {}'.format(par.noise_scale_arr[3]))

# Label the plot, set plot title, set legend
plt.legend(loc = 'best')
plt.xlabel("Angular velocity [px/s]")
plt.ylabel("Average model-centroid point difference [px]")
plt.title("Meteor angle {} [deg]".format(par.phi))


# Configure the plot axis
plt.axis('tight')

plt.xlim((min(omega_arr), max(omega_arr)))
plt.ylim((min(noise0_data), max(noise0_data)))

# Save and show plot
plt.savefig('../Graphs/ODN/graph_odn_rolling.png')
plt.show()