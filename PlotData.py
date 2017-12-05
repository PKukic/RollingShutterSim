import matplotlib.pyplot as plt
import numpy as np


# Loading .NPZ file
data = np.load('data.npz')

# Setting variable names
omega_arr = data['arr_0']
noise0 = data['arr_0']
noise1 = data['arr_1']
noise2 = data['arr_2']

print(noise0)
print(noise1)
print(noise2)

# Plotting
#plt.scatter(omega_arr, noise0, c = 'red')
plt.scatter(omega_arr, noise1, c = 'green')
plt.scatter(omega_arr, noise2, c = 'blue')
plt.xlabel('Angular velocity [deg/s]')
plt.ylabel('Average distance of meteor and centroid points')
plt.xlim([0, np.amax(omega_arr)])

# Saving figure
plt.savefig('noise_difference_graph.png')

plt.show()
