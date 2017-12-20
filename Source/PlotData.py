""" Plotting data obtained by the simulation.
"""


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

# Loading .NPZ file
data = np.load('data_phi_frame_diff_rolling.npz')

# Setting array names
pp = data['arr_0']
ff = data['arr_1']
diff_data = data['arr_2']

# Set plot
fig, ax = plt.subplots()
plot = ax.pcolor(pp, ff, diff_data.T, cmap = 'inferno')
cbar = plt.colorbar(plot)

# Labels
plt.xlabel('Meteor angle $\phi$ [deg]')
plt.ylabel('Frame number')
cbar.ax.set_ylabel('Distance from model point [px]')

# Axis configuration
plt.axis('tight')

# Saving plot
plt.savefig('graph_phi_frame_diff_rolling')

plt.show()