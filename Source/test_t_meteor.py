import SimulationTools as st
import Parameters as par
import matplotlib.pyplot as plt
import numpy as np

phi_arr = np.arange(1, 361, 5)

for phi in phi_arr:

	show_plots = False

	t_meteor, end_coordinates = st.timeFromAngle(phi, par.omega, par.img_x, par.img_y, par.scale, par.fps)
	print("{:.4f}".format(t_meteor))

	model_coordinates, centroid_coordinates = st.pointsCentroidAndModel(par.rolling_shutter, t_meteor, phi, \
	            par.omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, par.noise_scale, par.offset, show_plots)

	plt.scatter(end_coordinates[0], end_coordinates[1])
	plt.scatter(par.img_x/2, par.img_y/2)
	plt.xlim([0, par.img_x])
	plt.ylim([0, par.img_y])

	plt.show()