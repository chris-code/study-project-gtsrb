import PIL.Image as pil
import numpy as np

def read_data(filename, resolution, n=None):
	'''This method takes a file containing the image paths and returns a 4D array containing the image data and a 1D array containing the labels.'''

	# open file which contains the image list
	with open(filename, "r") as f:
		# read image paths from file and safe them in a list
		image_paths = f.readlines()

		# check whether there is a limit for the images to be loaded
		first_dim_size = n if n is not None else len(image_paths)

		# create empty arrays with appropriate size
		X_train = np.empty((first_dim_size, 3, resolution[0], resolution[1]))
		y_train = np.empty((first_dim_size))

		# iterate over all images
		for idx, image_path in enumerate(image_paths):
			# break after having processed the desired number of images
			if idx >= n:
				break

			# read line until the space character, which separates the path from the label
			im = pil.open(image_path[0:image_path.find(" ")])

			# resize image so desired size
			im = im.resize(size)

			# save image as array within the result array
			X_train[idx] = np.transpose(np.asarray(im), [2,0,1])
			
			# save label
			y_train[idx] = int(image_path[image_path.find(" "):])

	return X_train, y_train

size = (48,48)
filename = "/home/yannickubuntu/workspaceStudienprojekt/study-project-gtsrb/data/image_list.txt"

X_train, y_train = read_data(filename, size, 20)
print(X_train.shape)
print(y_train.shape)
print (y_train)
