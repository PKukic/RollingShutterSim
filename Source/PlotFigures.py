import SimulationTools as st 
import Parameters as par 

omega = 40
phi = 45
noise_scale = 10

# Calculate duration of the meteor
t_meteor = st.timeFromAngle(phi, omega, par.img_x, par.img_y, par.scale, par.fps)
print("Meteor duration: {:.2f}".format(t_meteor))

fit_param = [0, 0]

# Simulate global shutter meteor
rolling_shutter = False
show_plots = False
time_coordinates, centroid_coordinates, model_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
   	omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise_scale, par.offset, fit_param, show_plots)

# Simulate rolling shutter meteor
rolling_shutter = True
show_plots = True
time_coordinates, centroid_coordinates, model_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
   	omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise_scale, par.offset, fit_param, show_plots)