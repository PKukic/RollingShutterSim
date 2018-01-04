""" Plotting data obtained by the simulation.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Meteor angle
phi = 45

# Loading .NPZ file
data = np.load('../Data/OYD-R/data_oyd_rolling_{}.npz'.format(phi))

# Setting array names
omega_arr = data['arr_0']
ycentr_data = data['arr_1']
diff_data = data['arr_2']

# Set plot
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(omega_arr, ycentr_data, diff_data, rstride = 1000, cstride = 1000)


# Legends and labels
ax.set_zlabel("Model-centroid  point difference [px]")
ax.set_xlabel("Angular velocity $\omega$ [px/s]")
ax.set_ylabel("Centroid Y coordinate [px]")
plt.title("Meteor angle fixed to: {} [deg]".format(phi))

# Configure axis
plt.axis('tight')

# plt.savefig("../Graphs/OYD-R/graph_oyd_3D_rolling_{}.png".format(phi))

plt.show()