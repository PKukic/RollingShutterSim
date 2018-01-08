""" Plot data obtained by the omega-difference-noise simulation, in which both rolling shutter
	and global shutter are used
"""

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import matplotlib.pyplot as plt
import numpy as np

# Loading data
data = np.load('../Data/ODN/data_odn_global_rolling.npz')

# Unpack daata and set array names
omega_arr = data['arr_0']
noise0_rolling = data['arr_1']
noise1_rolling = data['arr_2']
noise2_rolling = data['arr_3']
noise3_rolling = data['arr_4']

noise0_global = data['arr_5']
noise1_global = data['arr_6']
noise2_global = data['arr_7']
noise3_global = data['arr_8']

# Plot four noise arrays for global shutter
plt.plot(omega_arr, noise0_global, 'c-', label = '$\sigma$ = 0, glob.')
plt.plot(omega_arr, noise1_global, 'r-', label = '$\sigma$ = 5, glob.')
plt.plot(omega_arr, noise2_global, 'g-', label = '$\sigma$ = 10, glob.')
plt.plot(omega_arr, noise3_global, 'b-', label = '$\sigma$ = 20, glob.')


# Plot four noise arrays for rolling shutter
plt.plot(omega_arr, noise0_rolling, 'c--', label = '$\sigma$ = 0, roll.')
plt.plot(omega_arr, noise1_rolling, 'r--', label = '$\sigma$ = 5, roll.')
plt.plot(omega_arr, noise2_rolling, 'g--', label = '$\sigma$ = 10, roll.')
plt.plot(omega_arr, noise3_rolling, 'b--', label = '$\sigma$ = 20, roll.')

# Set plot legends and labels, set plot title
plt.legend(loc = 'lower right')
plt.xlabel("Angular velocity [deg/s]")
plt.ylabel("Average model-centroid point difference [px]")
plt.title("Meteor angle 45 [deg]")

# Configure the plot axis
plt.axis('tight')

# Save and show plot
plt.savefig('../Graphs/ODN/graph_odn_global_rolling.png')
plt.show()