import numpy as np
import keras.callbacks
import scipy.ndimage.interpolation as trans

class Distortions(keras.callbacks.Callback):
	def __init__(self, x, number_of_images):
		self.original_x = np.copy(x)
		self.x = x
		self.resolution = x.shape[2:4]
		self.number_of_images = number_of_images

	def cut_image(self, image):
		# ATTENTION! It is assumed that the images are square!

		# image has a smaller resolution than desired
		if image.shape[1] < self.resolution[0]:
			# initialize result image
			res = np.empty((3,self.resolution[0],self.resolution[1]))

			# calculate boundaries
			left = (self.resolution[0] - image.shape[1]) // 2
			top = (self.resolution[1] - image.shape[2]) // 2
			right = left + image.shape[1]
			bottom = top + image.shape[2]

			res[:,left:right,top:bottom] = image

			# interpolate empty pixels (left, right, above, below distorted image)
			res[:,:left,top:bottom] = image[:,0,:].reshape(3,1,-1)
			res[:,right:,top:bottom] = image[:,-1,:].reshape(3,1,-1)
			res[:,left:right,:top] = image[:,:,0].reshape(3,-1,1)
			res[:,left:right,bottom:] = image[:,:,-1].reshape(3,-1,1)

			# interpolate empty pixels (topleft, topright, bottomleft, bottomright corner)
			#res[:,:left,:top] = image[:,0,0].resize(res[:,:left,:top].shape)
			#res[:,right:,:top] = image[:,-1,0].resize(res[:,right:,:top].shape)
			#res[:,:left,bottom:] = image[:,0,-1].resize(res[:,:left,bottom:].shape)
			#res[:,right:,bottom:] = image[:,-1,-1].resize(res[:,right:,bottom:].shape)

			#res[:,:left,:top] = np.empty(res[:,:left,:top].shape).fill(image[:,0,0])
			#res[:,right:,:top] = np.empty(res[:,right:,:top].shape).fill(image[:,-1,0])
			#res[:,:left,bottom:] = np.empty(res[:,:left,bottom:].shape).fill(image[:,0,-1])
			#res[:,right:,bottom:] = np.empty(res[:,right:,bottom:].shape).fill(image[:,-1,-1])

			print("foo" + " " + str(left) + " " + str(top))
			print(trans.zoom(image[:,0,0].reshape(3,1,1), zoom=[1,left,top]).shape)
			print("bar")

			res[:,:left,:top] = trans.zoom(image[:,0,0].reshape(3,1,1), zoom=[1,left,top])
			res[:,right:,:top] = trans.zoom(image[:,-1,0].reshape(3,1,1), zoom=[1,self.resolution[0]-right,top])
			res[:,:left,bottom:] = trans.zoom(image[:,0,-1].reshape(3,1,1), zoom=[1, left, self.resolution[1]-bottom])
			res[:,right:,bottom:] = trans.zoom(image[:,-1,-1].reshape(3,1,1), zoom=[1, self.resolution[0]-right, resolution[1]-bottom])

			return res

		# image has a larger resolution than desired
		else:
			# calculate boundaries
			left = (image.shape[1] - self.resolution[0]) // 2
			top = (image.shape[2] - self.resolution[1]) // 2
			right = left + self.resolution[0]
			bottom = top + self.resolution[1]

			# return result image
			return image[:,left:right,top:bottom]

	def on_epoch_begin(self, epoch, logs={}):
		# create random values for shift, rotation and scaling
		rotate_angles = np.random.uniform(-5., 5., (self.number_of_images))
		scale_factors = np.random.uniform(0.9, 1.1, (self.number_of_images))
		shift_values = np.random.uniform(- 0.1 * self.x.shape[2], 0.1 * self.x.shape[2], (self.number_of_images))

		# iterate over all images and transform them
		for img_id in range(self.number_of_images):
			# shift
			img = trans.shift(self.original_x[img_id], [0, shift_values[img_id], shift_values[img_id]], mode="nearest")

			# rotate
			img = trans.rotate(img, axes=(1,2), angle=rotate_angles[img_id], reshape=False, mode="nearest")

			# scale
			img = trans.zoom(img, zoom=[1,scale_factors[img_id],scale_factors[img_id]], mode="nearest")

			self.x[img_id] = self.cut_image(img)
