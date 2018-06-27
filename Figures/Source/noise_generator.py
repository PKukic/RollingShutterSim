import numpy as np 
import matplotlib.pyplot as plt 

img_y = img_x = 200
noise_scale = [0.01, 5, 10, 20]
offset = 50

fig, axes = plt.subplots(ncols=len(noise_scale))

for i, ax in zip(range(len(noise_scale)), axes):

	ax.imshow(offset + np.abs(np.random.normal(loc=0, scale=noise_scale[i], size=(img_y, img_x))), cmap='gray', vmin=0, vmax=255)
	
	ax.set_title(r"$\sigma$ = {}".format(noise_scale[i] if noise_scale[i] != 0.01 else 0))

	ax.axis("off")


plt.tight_layout()
plt.subplots_adjust(wspace=0.1)
plt.show()