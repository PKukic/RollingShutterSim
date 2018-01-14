""" Plotting data obtained by the omega-phi-average difference [px] rolling shutter simulation.
"""

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load data
data = np.load('../Data/data_nopgd_rolling.npz')

# Unpack and set array names
omega_arr = data['arr_0']
phi_arr = data['arr_1']
avg_diff_arr = data['arr_2']


### Filter outliers ###

def filter():
	global phi_arr
	global omega_arr
	global avg_diff_arr

	del_arr = []

	# Find outliers (phi == 360/0 deg)
	for i in range(len(phi_arr)):
		if phi_arr[i] == 0 or phi_arr[i] == 360 or phi_arr[i] == 180:
			del_arr.append(i)

	# Delete outliers
	omega_arr = np.delete(omega_arr, del_arr)
	phi_arr = np.delete(phi_arr, del_arr)
	avg_diff_arr = np.delete(avg_diff_arr, del_arr)


filter()

### Plot filtered data ###

# Set 3D scatter plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(omega_arr, phi_arr, avg_diff_arr, c = avg_diff_arr, cmap = 'inferno', lw = 0)

# Set plot legends and labels
ax.set_xlabel('$\omega_r$ [px/s]')
ax.set_ylabel('$\phi$ [px/s]')
ax.set_zlabel('$\Delta$r [px]')

# Configure axis
plt.axis('tight')

# Show plot
plt.show()