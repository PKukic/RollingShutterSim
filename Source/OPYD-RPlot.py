""" Plot data obtained by the meteor velocity [omega] - meteor angle [phi] - Y centroid coordinate [ycentr] - model-centroid point difference [diff] simulation
"""

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import
import matplotlib.pyplot as plt
import numpy as np
import os, os.path


# Getting number of files in the data directory
DIR = '../Data/OPYD-R'
file_num_range = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

# Checking the number of the files
print(file_num_range)

# Generate 2 plots for each file
for file_n in range(file_num_range):

	# Load data
	data = np.load('../Data/OPYD-R/data_opyd_rolling{}.npz'.format(file_n))

	# Unpack data and set array names
	omega_arr = data['arr_0']
	phi_data = data['arr_1']
	ycentr_data = data['arr_2']
	diff_data = data['arr_3']


	### 2D contour graph; color = diff ###

	# Disable interactive mode - no showing plots
	plt.ioff()

	# Set 2D color plot
	fig, ax = plt.subplots()
	plot = ax.scatter(phi_data, ycentr_data, c=diff_data, cmap = 'inferno', lw = 0)
	cbar = plt.colorbar(plot)

	# Set plot legends and labels
	cbar.set_label("Model-centroid  point difference [px]")
	ax.set_xlabel("Meteor angle $\phi$ [deg]")
	ax.set_ylabel("Centroid Y coordinate [px]")
	plt.title('Angular velocity: {:.2f} [px/s]'.format(omega_arr[file_n]))

	# Configure axis
	plt.axis('tight')

	# Save the plot and close it
	plt.savefig("../Graphs/OPYD-R/graph_opyd_rolling{}.png".format(file_n))
	plt.close()

	### 2D color plot; color = phi ###

	# Set 2D color plot with different axes
	fig, ax = plt.subplots()
	plot = ax.scatter(diff_data, ycentr_data, c=phi_data, cmap = 'inferno', lw = 0)
	cbar = plt.colorbar(plot)

	# Set plot legends and labels
	cbar.set_label("Meteor angle $\phi$ [deg]")
	ax.set_xlabel("Model-centroid  point difference [px]")
	ax.set_ylabel("Centroid Y coordinate [px]")
	plt.title('Angular velocity: {:.2f} [px/s]'.format(omega_arr[file_n]))

	# Configure axis
	plt.axis('tight')

	# Save the plot and close it
	plt.savefig("../Graphs/ODYP-R/graph_odyp_rolling{}.png".format(file_n))
	plt.close()