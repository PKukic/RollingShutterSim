""" Plot data obtained by the omega-ycentr-diff simulation
"""

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import matplotlib.pyplot as plt
import numpy as np

# Fixed meteor angle (parameters that change are the meteor's velocity and the Y coordinate of the centroid point)
phi = 45

# Load data
data = np.load('../Data/OYD-R/data_oyd_rolling_{}.npz'.format(phi))

# Unpack and set array names
omega_arr = data['arr_0']
ycentr_data = data['arr_1']
diff_data = data['arr_2']

# Set 2D color plot
fig, ax = plt.subplots()
plot = ax.scatter(omega_arr, ycentr_data, c = diff_data, cmap = 'inferno', lw = 0)
cbar = plt.colorbar(plot)

# Set plot legends and labels, set plot title
cbar.set_label("Model-centroid  point difference [px]")
ax.set_xlabel("Angular velocity $\omega$ [px/s]")
ax.set_ylabel("Centroid Y coordinate [px]")
plt.title("Meteor angle fixed to: {} [deg]".format(phi))

# Configure axis
plt.axis('tight')

# Save the plot and show it
plt.savefig("../Graphs/OYD-R/graph_oyd_rolling_{}.png".format(phi))
plt.show()