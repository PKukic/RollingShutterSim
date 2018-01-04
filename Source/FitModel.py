import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt

# Fixed meteor angle
phi = 45

# Loading data
data = np.load('../Data/OYD-R/data_oyd_rolling_{}.npz'.format(phi))

# Setting variable names
omega = data['arr_0']
ycentr = data['arr_1']
diff = data['arr_2']

# Model function

def rollingModel(par, a, b, c):

	# Extracting parameters from tuple
	omega = par[0]
	ycentr = par[1]

	# Model
	return a * omega + b * ycentr ** 2 + c * ycentr


# Fitting
param, pcov = opt.curve_fit(rollingModel, (omega, ycentr), diff)
print(param)

# Checking fit
delta = diff - rollingModel((omega, ycentr), *param)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(omega, ycentr, delta, lw = 0)

ax.set_xlabel("Angular velocity $\omega$ [px/s]")
ax.set_ylabel("Centroid Y coordinate [px]")
ax.set_zlabel("Model - data")
plt.title("Meteor angle fixed to: {} [deg]".format(phi))

plt.axis('tight')

plt.show()