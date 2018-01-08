""" Plotting data obtained by the simulation.
"""
# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Loading NPZ file
data = np.load('../Data//Velocity error/data_velocity_error3D.npz')

# Naming arrays for plotting
omega_arr = data['arr_0']
phi_arr = data['arr_1']
deltav_arr = data['arr_2']

# Set plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(omega_arr, phi_arr, deltav_arr, c = deltav_arr, cmap = 'inferno', lw = 0)

# Legends and labels
ax.set_xlabel('$\omega$ [px/s]')
ax.set_ylabel('$\phi$ [deg]')
ax.set_zlabel('$\Delta$v [px/s]')

# Configure axis
plt.axis('tight')

plt.show()