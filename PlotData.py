""" Plotting data obtained by the simulation.
"""


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

# Loading .NPZ file
data = np.load('data_frame_angle_diff_rolling.npz')

# Setting variable names
phi_data = data['arr_0']
frame_num_data = data['arr_1']
diff_data = data['arr_2']

ax = plt.axes(projection='3d')
ax.scatter3D(phi_data, frame_num_data, diff_data)
plt.show()

"""
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
"""