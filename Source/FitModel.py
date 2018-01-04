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

def rollingModel(par, a, b):

	# Extracting parameters from tuple
	omega = par[0]
	ycentr = par[1]

	# Model
	return (a - b * ycentr) * omega


# Fitting
param, pcov = opt.curve_fit(rollingModel, (omega, ycentr), diff)
print(param)

# Checking fit
fit = rollingModel((omega, ycentr), *param)
delta = diff - rollingModel((omega, ycentr), *param)

### Delta plot ###

# Set plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(omega, ycentr, delta, lw = 0)

# Labels
ax.set_xlabel("Angular velocity $\omega$ [px/s]")
ax.set_ylabel("Centroid Y coordinate [px]")
ax.set_zlabel("Difference between model and data")
plt.title("Meteor angle fixed to: {} [deg]".format(phi))

# Configure axis
plt.axis('tight')

plt.savefig('../Graphs/Model graphs/graph_model_1_delta.png')

plt.show()


### Fit and data plot ###

# Set plot
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(omega, ycentr, fit, rstride = 1000, cstride = 1000)
ax.scatter(omega, ycentr, diff, c = diff, lw = 0)

# Labels
ax.set_zlabel("Model-centroid  point difference [px]")
ax.set_xlabel("Angular velocity $\omega$ [px/s]")
ax.set_ylabel("Centroid Y coordinate [px]")
plt.title("Meteor angle fixed to: {} [deg]".format(phi))

# Configure axis
plt.axis('tight')

plt.savefig('../Graphs/Model graphs/graph_model_1_rep.png')

plt.show()