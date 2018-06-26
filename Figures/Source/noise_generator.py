import numpy as np 
import matplotlib.pyplot as plt 

img_y = img_x = 1000
noise_scale = [0.01, 5, 10, 20]

for i in range(len(noise_scale)):

	plt.imshow(np.random.normal(loc=0, scale=noise_scale[i], size=(img_y, img_x)), cmap='gray', vmin=0, vmax=255)
	plt.title(r"$\sigma$ = {}".format(noise_scale[i] if noise_scale[i] != 0.01 else 0))
	plt.axis("off")
	plt.show()