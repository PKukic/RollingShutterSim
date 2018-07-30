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

shut_name = [interlaced, rolling, gshutter]
name_arr = ['Global shutter', 'Interlaced shutter', 'Rolling shutter']

fig, axes = plt.subplots(ncols=3)

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

for i, ax in zip(range(3), axes):
	vert_arr = shut_name[i]
	vert_arr = np.repeat(vert_arr, y_size*x_size/y_bands)
	vert_arr = vert_arr.reshape(y_size, x_size)

	x_p = np.linspace(0, x_size, x_size)
	y_p = np.linspace(0, y_size, y_size)
	X, Y = np.meshgrid(x_p, y_p)

	ax.pcolor(X, Y, vert_arr, cmap='viridis_r')
	ax.axis('off')
	plt.gca().xaxis.set_major_locator(matplotlib.ticker.NullLocator())
	plt.gca().yaxis.set_major_locator(matplotlib.ticker.NullLocator())


	name = name_arr[i]
	ax.set_title(name)



plt.tight_layout()
plt.subplots_adjust(wspace=0.1)

# plt.figure(figsize = (16, 9))

# plt.savefig('../Images/shutters/collage_shutters.pdf')
# plt.savefig('../Images/shutters/collage_shutters.png')

plt.show()