""" Plotting data obtained by the simulation.
"""
# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import matplotlib.pyplot as plt
import numpy as np

# Loading .NPZ file
data = np.load('../Data/data_odn_global_rolling.npz')

# Assign array names to .NPZ arrays
omega_arr = data['arr_0']
noise0_global = data['arr_1']
noise1_global = data['arr_2']
noise2_global = data['arr_3']
noise3_global = data['arr_4']

noise0_rolling = data['arr_5']
noise1_rolling = data['arr_6']
noise2_rolling = data['arr_7']
noise3_rolling = data['arr_8']

# Plot four noise arrays for global shutter
plt.plot(omega_arr, noise0_global, 'c-', label = '$\sigma$ = 0, glob.')
plt.plot(omega_arr, noise0_global, 'c-', label = '$\sigma$ = 0, glob.')
plt.plot(omega_arr, noise0_global, 'c-', label = '$\sigma$ = 0, glob.')
plt.plot(omega_arr, noise0_global, 'c-', label = '$\sigma$ = 0, glob.')


# Plot four noise arrays for rolling shutter
plt.plot(omega_arr, noise0_rolling, 'c--', label = '$\sigma$ = 0, roll.')
plt.plot(omega_arr, noise0_rolling, 'c--', label = '$\sigma$ = 0, roll.')
plt.plot(omega_arr, noise0_rolling, 'c--', label = '$\sigma$ = 0, roll.')
plt.plot(omega_arr, noise0_rolling, 'c--', label = '$\sigma$ = 0, roll.')

# Legends and labels
plt.legend(loc = 'lower right')
plt.xlabel("Angular velocity [deg/s]")
plt.ylabel("Average model-centroid point difference [px]")
plt.title("Meteor angle 45 [deg]")

# Configuring axis
plt.axis('tight')

plt.savefig('../Graphs/graph_odn_global_rolling.png')

plt.show()
