""" Plotting data obtained by the simulation.
"""
# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import matplotlib.pyplot as plt
import numpy as np

### ODN plots ###

# Loading NPZ file
data = np.load('../Data/ODN/data_odn_global.npz')

# Assign variables to .NPZ arrays
omega_arr = data['arr_0']
noise0_arr = data['arr_1']
noise5_arr = data['arr_2']
noise10_arr = data['arr_3']
noise20_arr = data['arr_4']

# Plot all four noise arrays
plt.plot(omega_arr, noise0_arr, 'c-', label = '$\sigma$ = 0')
plt.plot(omega_arr, noise5_arr, 'r-', label = '$\sigma$ = 5')
plt.plot(omega_arr, noise10_arr, 'g-', label = '$\sigma$ = 10')
plt.plot(omega_arr, noise20_arr, 'b-', label = '$\sigma$ = 20')

# Legends and labels
plt.legend(loc = 'lower right')
plt.xlabel("Angular velocity [deg/s]")
plt.ylabel("Average model-centroid point difference [px]")
plt.title("Meteor angle 45 [deg]")

# Configuring axis
plt.axis('tight')

plt.savefig('../Graphs/ODN/graph_odn_global2.png')

plt.show()


