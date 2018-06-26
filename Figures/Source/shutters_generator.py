import matplotlib
import matplotlib.pyplot as plt
import numpy as np

x_size = 100
y_size = 100

y_bands = 10

interlaced = np.array([1, 0] * (y_bands/2))
rolling = np.linspace(1, y_bands, y_bands)
gshutter = np.array([0] * y_bands)

print(interlaced)
print(rolling)
print(gshutter)

def draw(x_size, y_size, y_bands, vert_arr, name):
	vert_arr = np.repeat(vert_arr, y_size*x_size/y_bands)
	vert_arr = vert_arr.reshape(y_size, x_size)

	x_p = np.linspace(0, x_size, x_size)
	y_p = np.linspace(0, y_size, y_size)
	X, Y = np.meshgrid(x_p, y_p)

	plt.pcolor(X, Y, vert_arr, cmap='viridis_r')
	plt.axis('off')
	plt.gca().xaxis.set_major_locator(matplotlib.ticker.NullLocator())
	plt.gca().yaxis.set_major_locator(matplotlib.ticker.NullLocator())
	plt.title(name)
	plt.savefig('../Figures/shutters/{}.png'.format(name), transparent = True, bbox_inches = 'tight', pad_inches = 0)
	#plt.show()

draw(x_size, y_size, y_bands, gshutter, 'Global shutter')
draw(x_size, y_size, y_bands, interlaced, 'Interlaced shutter')
draw(x_size, y_size, y_bands, rolling, 'Rolling shutter')
