import SimulationTools as st
import Parameters as par
import numpy as np 
import matplotlib.pyplot as plt


start = 0
fin = 11
step = 1
delta_omega = np.arange(start, fin, step)


a = 1
omega_start = 20
b_arr = []

fin_t = 0.5
dt = 0.005
time_arr = np.arange(0, fin_t, dt)

for i in range(len(delta_omega)):
	b_arr.append(st.getparam(a, omega_start, omega_start - delta_omega[i], fin_t))


omega_arr = []

for i in range(len(delta_omega)):

	b_iter = b_arr[i]
	omega_arr_iter = []

	for t in time_arr:

		omega_arr_iter.append((omega_start - a*b_iter*np.exp(b_iter*t))*par.scale)

	omega_arr.append(omega_arr_iter)


for i in range(len(delta_omega)):

	plt.plot(time_arr, omega_arr[i], label = r'$\Delta \omega = {}$ [deg/s]'.format(delta_omega[i]))


plt.title('Meteor deceleration')
plt.xlabel('Time [s]')
plt.ylabel('Angular velocity [deg/s]')
plt.legend(loc = 'best')

plt.savefig('../Figures/Images/deceleration/dec_rep.png')

plt.show()