""" Plotting data obtained by the omega-phi-delta velocity rolling shutter simulation.
"""

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load data
data = np.load('../Data/OPdV-R/data_velocity_error3D.npz')

# Unpack and set array names
omega_arr = data['arr_0']
phi_arr = data['arr_1']
deltav_arr = data['arr_2']

# Set 3D scatter plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(omega_arr, phi_arr, deltav_arr, c = deltav_arr, cmap = 'inferno', lw = 0)

# Set plot legends and labels
ax.set_xlabel('$\omega$ [px/s]')
ax.set_ylabel('$\phi$ [deg]')
ax.set_zlabel('$\Delta$v [px/s]')

# Configure axis
plt.axis('tight')

# Show plot
plt.show()