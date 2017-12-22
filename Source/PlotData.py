""" Plotting data obtained by the simulation.
"""


import matplotlib.pyplot as plt
import numpy as np
import os, os.path
from mpl_toolkits import mplot3d

file_num_range = len([name for name in os.listdir('.') if os.path.isfile(name)])

print(file_num_range)

for file_n in range(file_num_range-1):

	# Loading .NPZ file
	data = np.load('../Data/APYD-R/data_apyd_rolling{}.npz'.format(file_n))

	# Setting array names
	phi_data = data['arr_0']
	ycentr_data = data['arr_1']
	diff_data = data['arr_2']


	### 2D contour graph ###

	# Set plot
	fig, ax = plt.subplots()
	plot = ax.scatter(phi_data, ycentr_data, c=diff_data, cmap = 'inferno')
	cbar = plt.colorbar(plot)

	# Legends and labels
	cbar.set_label("Model-centroid point difference")
	ax.set_xlabel("Meteor angle $\phi$ [deg]")
	ax.set_ylabel("Centroid Y coordinate")

	# Configure axis
	plt.axis('tight')

	# Saving color plot
	plt.savefig("../Graphs/APYD-R/graph_apyd_rolling{}.png".format(file_n))

	plt.show()

	### 3D plot ### 
	"""
	# Set plot
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.scatter3D(phi_data, ycentr_data, diff_data)

	# Legends and labels
	ax.set_xlabel("Meteor angle $\phi$")
	ax.set_ylabel("Centroid Y coordinate")
	ax.set_zlabel("Model-centroid point difference")

	# Saving 3D plot
	plt.savefig("../Graphs/3D_phi_ycentr_diff_rolling.png")

	plt.show()
	"""