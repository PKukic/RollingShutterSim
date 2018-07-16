import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt 
import Parameters as par
import SimulationTools as st 

data = np.load('../Data/ODN/data_worst_case.npz')
omega_arr = [x*par.scale for x in data['arr_0']][:-14]
diff_arr = data['arr_1'][:-14]

# print(diff_arr[:-14])


# plt.plot(omega_arr, diff_arr)
# plt.show()

def linFit(x, a, b):
	return a*x + b

def proportional(x, a):
	return a*x

param_ab, pcov = opt.curve_fit(linFit, omega_arr, diff_arr)
param_a, pcov = opt.curve_fit(proportional, omega_arr, diff_arr)


a1 = param_ab[0]
a2 = param_a[0]
b = param_ab[1]

print('{:.5f} {:.5f}'.format(a1, b))

print(par.img_y)