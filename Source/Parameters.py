''' A file that contains the simulation parameters.
'''


import numpy as np

# Using rolling shutter
rolling_shutter = True

# Meteor duration
t_meteor = 0.5

# Meteor angle counterclockwise from the Y axis (deg)
phi = 45

# Meteor angle array
phi_array = np.arange(0, 361)

# Image size
img_x = 1280
img_y = 720


# Angle of the image diagonal [rad]
diag_angle = np.arctan(img_y / img_x)

# Pixel scale in px/deg
scale = img_x/42

#  Number of frames per second
fps = 25

# Meteor's angular velocity (deg/s)
omega = 50

# Angular velocity array in px/s #(logarithmic)
omega_pxs = np.logspace(np.log10(30), np.log10(1500), 10)

# Angular velocity array in deg/s #(logarithmic)
omega_arr = omega_pxs / scale

# Omega arr for ODN-G
omega_odn_arr = np.arange(1, 51, 0.5)

# Omega arr for OYD-R
omega_oyd_arr = np.arange(1, 50.25, 0.25)

# Standard deviation along X and Y axis
sigma_x = 2
sigma_y = 2

# Scale of background noise
noise_scale = 0

# Scale of background noise array
noise_scale_arr = [0, 5, 10, 20]

# Level of background offset
offset = 20

# Plot individual frames?
show_plots = False

# List of all unified parameters
param = [rolling_shutter, t_meteor, phi, omega, img_x, img_y, scale, fps, sigma_x, sigma_y, noise_scale, offset, show_plots]
