import SimulationTools as st 
import Parameters as par 

omega = 50
phi = 45
show_plots = True
noise_scale = 10

# Calculate duration of the meteor
t_meteor = st.timeFromAngle(phi, omega, par.img_x, par.img_y, par.scale, par.fps)
print("Meteor duration: {:.2f}".format(t_meteor))

fit_param = [0, 0]

# Simulate global shutter meteor
rolling_shutter = False
time_coordinates, centroid_coordinates, model_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
   	omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise_scale, par.offset, par.fit_param, show_plots)

# Simulate rolling shutter meteor
rolling_shutter = True
time_coordinates, centroid_coordinates, model_coordinates = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, \
   	omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, noise_scale, par.offset, par.fit_param, show_plots)