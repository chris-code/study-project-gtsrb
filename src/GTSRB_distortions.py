import numpy as np
import keras.callbacks
import scipy.ndimage.interpolation as trans

from scipy.misc import toimage # DELETE ME

class Distortions(keras.callbacks.Callback):
	def __init__(self, x, number_of_images):
		self.original_x = np.copy(x)
		self.resolution = x.shape[2:4]
		self.number_of_images = number_of_images
	
	def on_epoch_end(self, epoch, logs={}):

		rotate_angles = np.random.uniform(-5., 5., (self.number_of_images))
		scale_factors = np.random.uniform(0.9., 1.1, (self.number_of_images))
		shift_values = np.random.uniform(- 0.1 * x.shape[2], 0.1 * x.shape[2], (self.number_of_images))

		for img_id in range(self.number_of_images):
			toimage(self.original_x[img_id]).show()

			# rotate			
			img = trans.rotate(self.original_x[img_id], axes=(1,2), angle=rotate_angles[img_id]., reshape="false")
			toimage(img).show()

			# scale
			img = trans.zoom(img, zoom=[1,shift_values[img_id],shift_values[img_id]])


if __name__ == "__main__":
	print("Tio estas malbona")
