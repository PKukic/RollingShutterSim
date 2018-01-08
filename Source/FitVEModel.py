import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt

# Loading .NPZ file
data = np.load('../Data/Velocity error/data_velocity_error3D.npz')

# Naming arrays
omega_arr = data['arr_0']
phi_arr = data['arr_1']
deltav_arr = data['arr_2']

def correctionModel(var, a, b, c, d):

	# Define parameters
	omega = var[0]
	phi = np.deg2rad(var[1])

	# Model
	return a * omega ** b * np.sin(phi + np.pi/2) + c * omega ** d

param, pcov = opt.curve_fit(correctionModel, (omega_arr, phi_arr), deltav_arr)
print(param)

# Checking plot
fit = correctionModel((omega_arr, phi_arr), *param)
delta = deltav_arr - fit

### Fit and data plot ###

print(max(abs(min(delta)), max(delta)))

# Set plot
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(omega_arr, phi_arr, fit, rstride = 1000, cstride = 1000)

# Labels
ax.set_xlabel('$\omega$ [px/s]')
ax.set_ylabel('$\phi$ [deg]')
ax.set_zlabel('$\Delta$v [px/s]')


#plt.savefig('../Graphs/Model graphs/graph_correction_model_rep.png')

plt.show()

### Delta plot ###
'''
# Set plot
plt.scatter(phi_array, delta, lw = 0, c = 'r')

# Labels
plt.xlabel('$\phi$ [deg]')
plt.ylabel('Model-fit difference [px/s]')
plt.title('Angular velocity fixed to: {} [deg/s]'.format(par.omega))

plt.savefig('../Graphs/Model graphs/graph_correction_model_delta.png')

plt.close()
'''