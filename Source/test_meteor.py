import SimulationTools as st 
import Parameters as par
import matplotlib.pyplot as plt


# parameters
phi = 30
omega = 50
show_plots = False
rolling_shutter = True

print("Getting time from angle...")
t_meteor = st.timeFromAngle(phi, par.omega, par.img_x, par.img_y, par.scale, par.fps)

print("Simulating meteor...")
time_coordinates, centroid_coordinates, model_coordinates = st.pointsCentroidAndModel(rolling_shutter, par.t_meteor, phi, \
    omega, par.img_x, par.img_y, par.scale, par.fps, par.sigma_x, par.sigma_y, par.noise_scale, par.offset, show_plots)

print("Correcting coordinates...")
centroid_coordinates = st.coordinateCorrection(t_meteor, centroid_coordinates, par.img_y, par.fps)

print("Done!")

r_arr = []
r_arr.append(0)

cnt = 0

for i in range(len(centroid_coordinates) - 1):
	r = st.centroidDifference(centroid_coordinates[i], centroid_coordinates[i + 1])
	r_arr.append(cnt + r)
	cnt += r
	print('({:.2f}, {:.2f}), R: {:.2f}'.format(centroid_coordinates[i][0], centroid_coordinates[i][1], r))


print(len(r_arr))
print(len(time_coordinates))

plt.plot(time_coordinates, r_arr, 'o')
plt.title('Phi: {}'.format(phi))
plt.xlabel('Time')
plt.ylabel('Distance')

plt.show()