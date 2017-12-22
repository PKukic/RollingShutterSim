""" Plotting data obtained by the simulation.
"""


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

# Loading .NPZ file
data = np.load('../Data/data_phi_ycentr_diff_rolling.npz')

# Setting array names
phi_data = data['arr_0']
ycentr_data = data['arr_1']
diff_data = data['arr_2']


### 2D contour graph ###

# Set plot
fig, ax = plt.subplots()
plot = ax.scatter(phi_data, ycentr_data, c=diff_data, cmap = 'inferno')
cbar = plt.colorbar(plot)

# Legends and labels
cbar.set_label("Model-centroid point difference")
ax.set_xlabel("Meteor angle $\phi$ [deg]")
ax.set_ylabel("Centroid Y coordinate")

# Configure axis
plt.axis('tight')

# Saving color plot
plt.savefig("../Graphs/2DColor_phi_ycentr_diff_rolling.png")

plt.show()

### 3D plot ### 

# Set plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(phi_data, ycentr_data, diff_data)

# Legends and labels
ax.set_xlabel("Meteor angle $\phi$")
ax.set_ylabel("Centroid Y coordinate")
ax.set_zlabel("Model-centroid point difference")

# Saving 3D plot
plt.savefig("../Graphs/3D_phi_ycentr_diff_rolling.png")

plt.show()