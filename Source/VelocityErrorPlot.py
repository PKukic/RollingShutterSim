import matplotlib.pyplot as plt
import numpy as np

# Loading .NPZ file
data = np.load('../Data/data_velocity_error.npz')

# Naming arrays for plotting
phi_array = data['arr_0']
deltav_arr = data['arr_1']

# Set plot
plt.plot(phi_array, deltav_arr, 'o')
plt.xlabel('Meteor angle $\phi$ [deg]')
plt.ylabel("Velocity error [px/s]")
plt.axis('tight')

plt.savefig('../Graphs/velocity_error_graph.png')

plt.show()