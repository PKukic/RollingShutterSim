import numpy as np 
import matplotlib.pyplot as plt 
import scipy.optimize as opt
import Parameters as par 

# Loading .NPZ file
data = np.load('../Data/data_velocity_error.npz')

# Naming arrays
phi_array = data['arr_0']
deltav_arr = data['arr_2']

def correctionModel(phi, a, b):

	# Convert angle measure to radians
	phi = np.deg2rad(phi)

	# Model
	return a * np.sin(phi + np.pi/2) + b

param, pcov = opt.curve_fit(correctionModel, phi_array, deltav_arr)
print(param)

# Checking plot
fit = correctionModel(phi_array, *param)
delta = deltav_arr - fit

### Fit and data plot ###

plt.ioff()

# Set plot
plt.scatter(phi_array, correctionModel(phi_array, *param), lw = 0, c = 'r')
plt.scatter(phi_array, deltav_arr, lw = 0, c = 'b')

# Labels
plt.xlabel('$\phi$ [deg]')
plt.ylabel('Velocity error [px/s]')
plt.title('Angular velocity fixed to: {} [deg/s]'.format(par.omega))

plt.savefig('../Graphs/Model graphs/graph_correction_model_rep.png')

plt.close()

### Delta plot ###

# Set plot
plt.scatter(phi_array, delta, lw = 0, c = 'r')

# Labels
plt.xlabel('$\phi$ [deg]')
plt.ylabel('Model-fit difference [px/s]')
plt.title('Angular velocity fixed to: {} [deg/s]'.format(par.omega))

plt.savefig('../Graphs/Model graphs/graph_correction_model_delta.png')

plt.close()