import PIL.Image as pil
import numpy as np
import time
import random

def read_data_from_imagelist(filename, resolution, n=None):
	'''This method takes a file containing the image paths and returns a 4D array containing the image data and a 1D array containing the labels.'''

	# open file which contains the image list
	with open(filename, "r") as f:
		# read image paths from file and safe them in a list
		image_paths = f.readlines()
		random.shuffle(image_paths)

		# check whether there is a limit for the images to be loaded
		first_dim_size = n if n is not None else len(image_paths)

		# create empty arrays with appropriate size
		X = np.empty((first_dim_size, 3, resolution[0], resolution[1]))
		y = np.empty((first_dim_size))

		# iterate over all images
		for idx, image_path in enumerate(image_paths):
			# break after having processed the desired number of images
			if idx >= first_dim_size:
				break

			# read line until the space character, which separates the path from the label
			im = pil.open(image_path[0:image_path.find(" ")])

			# resize image so desired size
			im = im.resize(size)

			# save image as array within the result array
			X[idx] = np.transpose(np.asarray(im), [2,0,1])
			
			# save label
			y[idx] = int(image_path[image_path.find(" "):])

	return X, y


if __name__ == "__main__":
	size = (48,48)
	filename = "/home/yannickubuntu/workspaceStudienprojekt/study-project-gtsrb/data/image_list.txt"

	start_time = time.time()
	X_train, y_train = read_data_from_imagelist(filename, size)
	end_time = time.time()

	print(X_train.shape)
	print(y_train.shape)
	print (y_train)

	print("Execution time: " + str(end_time - start_time) + "s")
