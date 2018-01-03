""" Plotting data obtained by the simulation.
"""

import matplotlib.pyplot as plt
import numpy as np
import Parameters as par


# Loading .NPZ file
data = np.load('../Data/OYD-R/data_oyd_rolling_90.npz')

# Setting array names
omega_arr = data['arr_0']
ycentr_data = data['arr_1']
diff_data = data['arr_2']

# Set plot
fig, ax = plt.subplots()
plot = ax.scatter(omega_arr, ycentr_data, c = diff_data, cmap = 'inferno', lw = 0)
cbar = plt.colorbar(plot)

# Legends and labels
cbar.set_label("Model-centroid  point difference [px]")
ax.set_xlabel("Angular velocity $\omega$ [px/s]")
ax.set_ylabel("Centroid Y coordinate [px]")
plt.title("Meteor angle fixed to: {} [deg]".format(par.phi))

# Configure axis
plt.axis('tight')

plt.savefig("../Graphs/OYD-R/graph_oyd_rolling_90.png")

plt.show()