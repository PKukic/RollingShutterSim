import matplotlib.pyplot as plt
import numpy as np


# Loading .NPZ file
data = np.load('data.npz')

# Setting variable names
omega_arr = data['arr_0']
noise0 = data['arr_1']
noise1 = data['arr_2']
noise2 = data['arr_3']
noise3 = data['arr_4']

# Plotting
plt.plot(omega_arr, noise0, 'c-', label = 'No noise')
plt.plot(omega_arr, noise1, 'r-', label = 'Noise level 5 $\sigma$')
plt.plot(omega_arr, noise2, 'g-', label = 'Noise level 10 $\sigma$')
plt.plot(omega_arr, noise3, 'b-', label = 'Noise level 20 $\sigma$')
plt.xlabel('Angular velocity [deg/s]')
plt.ylabel('Average distance of meteor and centroid points')
plt.legend(loc = 'lower right')
plt.xlim([0, np.amax(omega_arr)])

# Saving figure
plt.savefig('noise_difference_graph.png')

plt.show()