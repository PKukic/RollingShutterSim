''' Fit a velocity error model to the meteor velocity [omega] - meteor angle [phi] - velocity error [deltav] data.
'''
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt

# Load data
data = np.load('../Data/OPdV-R/data_velocity_error3D.npz')

# Unpack and set array names
omega_arr = data['arr_0']
phi_arr = data['arr_1']
deltav_arr = data['arr_2']

# Model function
def correctionModel(par, c):

    Y_SIZE = 720
    FPS = 25

    # Define parameters and unpack from tuple
    omega = par[0]
    phi = np.deg2rad(par[1])

    a = -1.0/(Y_SIZE*FPS)
    b = -1.0/(Y_SIZE*FPS**2)

    return (a*omega**2)*np.sin(phi + np.pi/2) + (b*omega**2)*np.sin(2*phi + np.pi/2) + b*omega**2


# Fit model to data
param, pcov = opt.curve_fit(correctionModel, (omega_arr, phi_arr), deltav_arr)
print(param)

# Calculate the difference between the model and the actual data
fit = correctionModel((omega_arr, phi_arr), *param)
delta = deltav_arr - fit

### Fit and data plot ###

# Check the maximum deviation of the model from the actual data 
deviation = max(abs(min(delta)), max(delta))
print('Maximum deviation from the data: {:.2f}'.format(deviation))

# Set the 3D wireframe plot displaying the model
fig = plt.figure()
ax = Axes3D(fig)
#ax.plot_wireframe(omega_arr, phi_arr, fit, rstride = 1000, cstride = 1000)
ax.plot_wireframe(omega_arr, phi_arr, delta, rstride = 1000, cstride = 1000)

# Label and set plot title
ax.set_xlabel('$\omega$ [px/s]')
ax.set_ylabel('$\phi$ [deg]')
ax.set_zlabel('$\Delta$v [px/s]')

# Show plot
plt.show()