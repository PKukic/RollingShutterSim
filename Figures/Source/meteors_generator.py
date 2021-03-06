import sys

sys.path.insert(0, "../../Source")

import SimulationTools as st 
import Parameters as par 

rolling_shutter = True
phi = 45

# Correction for the new definition of phi
phi = st.ConvToSim(phi)


omega = 45
show_plots = True
noise_scale = 10

t_meteor = st.timeFromAngle(phi, omega, par.img_x, par.img_y, par.scale, par.fps)

# Get model and centroid coordinates
time_coordinates, centroid_coordinates, model_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
	omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise_scale, par.offset, par.fit_param, show_plots)