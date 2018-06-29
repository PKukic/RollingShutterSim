import sys

sys.path.insert(0, "../../Source")

import SimulationTools as st 
import Parameters as par 

rolling_shutter = False
phi = 135
omega = 45
show_plots = True
noise_scale = 10

# Get model and centroid coordinates
time_coordinates, centroid_coordinates, model_coordinates = st.pointsCentroidAndModel(rolling_shutter, par.t_meteor, phi, \
	omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise_scale, par.offset, par.fit_param, show_plots)