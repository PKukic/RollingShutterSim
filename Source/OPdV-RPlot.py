import matplotlib.pyplot as plt
import numpy as np

# Load data
data = np.load('../Data/data_velocity_error.npz')

# Unpack data and set array names
phi_array = data['arr_0']
deltav_avg_arr = data['arr_1']
deltav_true_arr = data['arr_2']

# Set plot
plt.plot(phi_array, deltav_avg_arr, 'bo', mew = 0, label = r'$\overline{v_g} - \overline{v_r}$')
plt.plot(phi_array, deltav_true_arr, 'ro', mew = 0, label = r'$\omega - \overline{v_r}$')
plt.xlabel('$\phi$ [deg]')
plt.ylabel("$\Delta$v [px/s]")
plt.legend(loc = 'upper right')

# Save the plot and show it
plt.savefig('../Graphs/OPdV-R/velocity_error_graph.png')
plt.show()