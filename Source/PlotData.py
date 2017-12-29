""" Plotting data obtained by the simulation.
"""


import matplotlib.pyplot as plt
import numpy as np
import os, os.path
# from mpl_toolkits import mplot3d

# Getting file number
DIR = '../Data/OPYD-R'
file_num_range = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

#Checking
print(file_num_range)

for file_n in range(file_num_range):

	# Loading .NPZ file
	data = np.load('../Data/OPYD-R/data_opyd_rolling{}.npz'.format(file_n))

	# Setting array names
	omega_arr = data['arr_0']
	phi_data = data['arr_1']
	ycentr_data = data['arr_2']
	diff_data = data['arr_3']


	### 2D contour graph ###

	# Set plot
	fig, ax = plt.subplots()
	plot = ax.scatter(phi_data, ycentr_data, c=diff_data, cmap = 'inferno', lw = 0)
	cbar = plt.colorbar(plot)

	# Legends and labels
	cbar.set_label("Model-centroid  point difference")
	ax.set_xlabel("Meteor angle $\phi$ [deg]")
	ax.set_ylabel("Centroid Y coordinate")
	plt.title('Angular velocity: {:.2f} [px/s]'.format(omega_arr[file_n]))

	# Configure axis
	plt.axis('tight')

	# Saving color plot
	plt.savefig("../Graphs/OPYD-R/graph_opyd_rolling{}.png".format(file_n))

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