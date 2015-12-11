import PIL.Image as pil
import numpy as np
import time
import random
import csv

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
			im = im.resize(resolution)

			# save image as array within the result array
			X[idx] = np.transpose(np.asarray(im), [2,0,1])
			
			# save label
			y[idx] = int(image_path[image_path.find(" "):])

	return X, y


def read_data(filename, resolution, n=None):
	'''This method takes a file containing the csv paths and returns a 4D array containing the image data and a 1D array containing the labels.'''

	# open file which contains the csv list
	with open(filename, "r") as f:
		# read csv paths from file and safe them in a list
		csv_paths = f.readlines()
		
		# create list, which is supposed to save information about the images
		image_list = []

		# iterate over csv files
		for csv_path in csv_paths:
			# remove possible new lines at the end of the path
			csv_path = csv_path.strip()

			# open csv file to read images
			with open(csv_path) as csv_file:
				# create csv reader
				csv_reader = csv.reader(csv_file, delimiter=";")

				# iterate over images
				for image_row in csv_reader:
					# build path to image
					image_path = csv_path[:csv_path.rfind('/')] + "/" + image_row[0]

					# append image to image list
					image_list.append({'path': image_path, 'corners':  (int(image_row[3]), int(image_row[4]), int(image_row[5]), int(image_row[6])), 'label': image_row[7]})

	# shuffle images
	random.shuffle(image_list)

	# check whether there is a limit for the images to be loaded
	number_of_images = n if n is not None else len(image_list)

	# create empty arrays with appropriate size
	X = np.empty((number_of_images, 3, resolution[0], resolution[1]))
	y = np.empty((number_of_images))

	# iterate over images
	for idx, image in enumerate(image_list):
		if idx >= number_of_images:
			break;
		
		# read line until the space character, which separates the path from the label
		im = pil.open(image['path'])

		# crop image
		im.crop(image['corners'])

		# resize image so desired size
		im = im.resize(resolution)
		
		# save image as array within the result array
		X[idx] = np.transpose(np.asarray(im), [2,0,1])

		# save label
		y[idx] = image['label']

	#print(image_list)

	return X, y

if __name__ == "__main__":
	size = (48,48)
	img_filename = "/home/yannickubuntu/workspaceStudienprojekt/study-project-gtsrb/data/image_list.txt"
	csv_filename = "/home/yannickubuntu/workspaceStudienprojekt/study-project-gtsrb/data/csv_list.txt"

	start_time = time.time()
	X_train, y_train = read_data(csv_filename, size, 10)
	end_time = time.time()

	print("Execution time: " + str(end_time - start_time) + "s")

	'''
	start_time = time.time()
	X_train, y_train = read_data_from_imagelist(img_filename, size)
	end_time = time.time()

	print(X_train.shape)
	print(y_train.shape)
	print (y_train)

	print("Execution time: " + str(end_time - start_time) + "s")
	'''
