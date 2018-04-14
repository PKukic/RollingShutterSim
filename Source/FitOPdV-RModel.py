''' Fit a velocity error model to the meteor velocity [omega] - meteor angle [phi] - velocity error [deltav] data.
'''
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
import Parameters as par

# Load data
data = np.load('../Data/OPdV-R/data_velocity_error_rolling.npz')

# Unpack and set array names
omega_arr = data['arr_0']
phi_arr = data['arr_1']
deltav_arr = data['arr_2']

# Model function
def correctionModel(param, a, b):

    # Define parameters and unpack from tuple
    omega = param[0]
    phi = np.deg2rad(param[1])

    # Model
    return (a*omega**2)*np.sin(phi + np.pi/2) + (b*omega**2)*np.sin(2*phi + np.pi/2) + b*omega**2


# Fit model to data
param, pcov = opt.curve_fit(correctionModel, (omega_arr, phi_arr), deltav_arr)
print(param)

# Compare with estimated parameters
a = -1.0/(par.img_y*par.fps)
b = -1.0/(par.img_y*par.fps**2)
print(a, b)

# Calculate the difference between the model and the actual data
fit = correctionModel((omega_arr, phi_arr), *param)
delta = deltav_arr - fit

# Check the maximum deviation of the model from the actual data 
deviation = max(abs(min(delta)), max(delta))
print('Maximum deviation from the data: {:.2f}'.format(deviation))

### Model - data difference plot ###

# Set 3D scatter plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(omega_arr, phi_arr, delta, c = delta, lw = 0)

# Label and set title
ax.set_xlabel("Angular velocity $\omega$ [px/s]")
ax.set_ylabel("Meteor angle [deg]")
ax.set_zlabel("Difference between model and data [px/s]")
plt.title('Model error')

# Configure plot axis
plt.axis('tight')

plt.show()


### Fit and data plot ###

# Set 3D plot of the model data represented as a wireframe,
# compare to 3D scatter plot of the actual data
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(omega_arr, phi_arr, fit, rstride = 1000, cstride = 1000)
ax.scatter(omega_arr, phi_arr, deltav_arr, c = deltav_arr, lw = 0)

# Label and set title
ax.set_xlabel("Angular velocity $\omega$ [px/s]")
ax.set_ylabel("Meteor angle [deg]")
ax.set_zlabel("Difference between model and data [px/s]")
plt.title('Model representation')

# Configure plot axis
plt.axis('tight')

plt.show()

