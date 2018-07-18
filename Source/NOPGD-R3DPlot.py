""" Plot data obtained by the noise-omega-phi-average difference [px] rolling shutter simulation.
"""

# Python 2/3 compatibility
from __future__ import print_function, division, absolute_import

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import Parameters as par

for noise in par.noise_scale_arr:

	print('Noise level: {}'.format(noise))

	# Load data
	data = np.load('../Data/TemporalNOPGD-R/data_tempnopgd_rolling_{}.npz'.format(noise))

	# Unpack and set array names
	omega_arr = data['arr_0']
	phi_arr = data['arr_1']
	avg_diff_arr = data['arr_2']

	### Plot data ###

	# Set 3D scatter plot
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(omega_arr, phi_arr, avg_diff_arr, c = avg_diff_arr, cmap = 'inferno', lw = 0)

	# Set plot legends and labels
	ax.set_xlabel('$\omega_r$ [px/s]')
	ax.set_ylabel('$\phi$ [deg]')
	ax.set_zlabel('$\Delta$r [px]')

	# Configure axis
	plt.axis('tight')

	plt.savefig('../Graphs/TemporalNOPGD-R/graph_temporal_nopgd_r_3D_{}'.format(noise))

	# Show plot
	plt.show()