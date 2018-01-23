''' A file that contains the simulation parameters.
'''

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import numpy as np

# Is a given simulation made using the rolling shutter camera?
rolling_shutter = True

# Meteor duration
t_meteor = 0.5

# Meteor angle counterclockwise from the Y axis (deg)
phi = 135

# Meteor angle array
phi_array = np.arange(0, 361)

# Image size
img_x = 1280
img_y = 720

# Pixel scale in px/deg
scale = img_x/42

#  Number of frames per second taken by the simulated camera
fps = 25

# Meteor's angular velocity (deg/s)
omega = 40

# Angular velocity array in px/s #(logarithmic)
omega_pxs = np.logspace(np.log10(30), np.log10(1500), 10)

# Angular velocity array in deg/s #(logarithmic)
omega_arr = omega_pxs / scale

# Meteor velocity array with step of 0.5
omega_odn_arr = np.arange(5, 50.5, 0.5)

# Meteor velocity array with step of 0.25
omega_oyd_arr = np.arange(1, 50.25, 0.25)

# Standard deviation along X and Y axis
sigma_x = 2
sigma_y = 2

# Scale of background noise
noise_scale = 0

# Scale of background noise array
noise_scale_arr = [0, 5, 10, 20]

# Temporary
del noise_scale_arr[:3]

print(noise_scale_arr)

# Level of background offset
offset = 20

# Plot individual frames?
show_plots = False

# List of all unified parameters
param = [rolling_shutter, t_meteor, phi, omega, img_x, img_y, scale, fps, sigma_x, sigma_y, noise_scale, offset, show_plots]
