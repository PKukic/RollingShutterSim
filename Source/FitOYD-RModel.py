`''' Fit a model to the meteor angle [omega] - Y centroid coordinate [ycentr] - Model-centroid point difference [diff] data.
'''

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt

# Fixed meteor angle (parameters that change are the meteor velocity and Y centroid coordinate)
phi = 45

# Load data
data = np.load('../Data/OYD-R/data_oyd_rolling_{}.npz'.format(phi))

# Unpack and set array names
omega = data['arr_0']
ycentr = data['arr_1']
diff = data['arr_2']

# Model function
def rollingModel(par, a, b):

	# Extract parameters from the tuple
	omega = par[0]
	ycentr = par[1]

	# Model
	return (a - b * ycentr) * omega


# Fit the model function to the data
param, pcov = opt.curve_fit(rollingModel, (omega, ycentr), diff)
print(param)

# Check the difference between the data and the model
fit = rollingModel((omega, ycentr), *param)
delta = diff - rollingModel((omega, ycentr), *param)

### Model - data difference plot ###

# Set 3D scatter plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(omega, ycentr, delta, c = delta, lw = 0)

# Label and set title
ax.set_xlabel("Angular velocity $\omega$ [px/s]")
ax.set_ylabel("Centroid Y coordinate [px]")
ax.set_zlabel("Difference between model and data")
plt.title("Model error; angle fixed to: {} [deg]".format(phi))

# Configure plot axis
plt.axis('tight')

plt.show()


### Fit and data plot ###

# Set 3D plot of the model data represented as a wireframe,
# compare to 3D scatter plot of the actual data
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(omega, ycentr, fit, rstride = 1000, cstride = 1000)
ax.scatter(omega, ycentr, diff, c = diff, lw = 0)

# Label and set title
ax.set_zlabel("Model-centroid  point difference [px]")
ax.set_xlabel("Angular velocity $\omega$ [px/s]")
ax.set_ylabel("Centroid Y coordinate [px]")
plt.title("Model representation; angle fixed to: {} [deg]".format(phi))

# Configure plot axis
plt.axis('tight')

plt.show()

### Notes ###
'''
parameter a = 1/FPS
parameter b = 1/(img_y * FPS)

final model ==> diff = (1 - ycentr/ysize) * (omega/fps)
'''