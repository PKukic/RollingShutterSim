import SimulationTools as st 
import Parameters as par

import matplotlib.pyplot as plt
import numpy as np

rolling_shutter = True
show_plots = False


phi = st.ConvToSim(par.phi)
omega = par.omega
t_meteor = st.timeFromAngle(phi, omega, par.img_x, par.img_y, par.scale, par.fps)


time_rolling, centroids_rolling, model_rolling = st.pointsCentroidAndModel(rolling_shutter, t_meteor, phi, omega, \
	par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, par.noise_scale, par.offset, par.fit_param, show_plots)


print(time_rolling)
print(centroids_rolling)

dist_rolling, velocity_rolling = st.getVelocity(time_rolling, centroids_rolling)
dist_model, velocity_model = st.getVelocity(time_rolling, model_rolling)

plt.plot(time_rolling, velocity_rolling, label = 'rolling')
plt.plot(time_rolling, velocity_model, label = 'model')

avg_avg = np.average([np.average(velocity_rolling), np.average(velocity_model)])
plt.ylim((avg_avg-100, avg_avg+100))

plt.xlabel('Time [s]')
plt.ylabel('Velocity [px/s]')

plt.legend(loc = 'best')

plt.savefig('/home/patrik/Desktop/t_vel.png')
plt.show()

plt.plot(time_rolling, dist_rolling, label = 'rolling')
plt.plot(time_rolling, dist_model, label = 'model')

plt.xlabel('Time [s]')
plt.ylabel('Distance from the starting centroid [px]')

plt.legend(loc = 'best')

plt.savefig('/home/patrik/Desktop/t_dist.png')
plt.show()