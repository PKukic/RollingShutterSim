import SimulationTools as st 
import numpy as np 
import matplotlib.pyplot as plt


start = 0
fin = 10
step = 1
delta_omega = np.arange(start, fin, step)


a = 1
omega_start = 50
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

		omega_arr_iter.append(omega_start + a*b_iter*np.exp(b_iter*t))

	omega_arr.append(omega_arr_iter)


for i in range(len(delta_omega)):

	plt.plot(time_arr, omega_arr[i], label = r'$\Delta \omega = {}$'.format(delta_omega[i]))


plt.title('Angular velocity [deg/s] with respect to time')
plt.xlabel('Time [s]')
plt.ylabel('Angular velocity [deg/s]')
plt.legend(loc = 'best')

plt.savefig('../Figures/Images/deceleration/dec_rep.png')

plt.show()