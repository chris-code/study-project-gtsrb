# -*- coding: utf-8 -*-

import PIL.Image as pil
import PIL.ImageOps as pilops
import numpy as np
import time
import random
import csv

def create_image_list(filename):
	'''This method takes the path to a .txt file, which contains the paths to the csv files (one per row), which contain the information about the images, which are to be loaded. The csv files contain the filename (with the path relative to the csv file), the width of the image, the height of the image, the coordinates X1, Y1, X2, Y2 of the bounding box (X1 and Y1 are inclusive, X2 and Y2 exclusive) and the class id of the image. Semicolons are used as seperators.'''
	# open file which contains the csv list
	with open(filename, "r") as f:
		# read csv paths from file and save them in a list
		csv_paths = f.readlines()

		# create list, which saves a dictionary for each image
		image_list = []

		# iterate over csv files
		for csv_path in csv_paths:
			# remove possible '\n' at the end of the path
			csv_path = csv_path.strip()

			# open csv file to read images
			with open(csv_path) as csv_file:
				# create csv reader
				csv_reader = csv.reader(csv_file, delimiter=";")

				# skip first line, which contains the headlines
				next(csv_reader)

				# save path to the folder, which contains the images
				folder_path = csv_path[:csv_path.rfind('/')] + "/"

				# iterate over images
				for image_row in csv_reader:
					# build path to image
					image_path = folder_path + image_row[0]

					# append image to image list
					image_list.append({'path': image_path, 'corners':  (int(image_row[3]), int(image_row[4]), int(image_row[5]), int(image_row[6])), 'label': image_row[7]})

	# shuffle images
	random.shuffle(image_list)

	return image_list

def count_classes(image_list):
	classes = set()
	for image in image_list:
		classes.add(image['label'])
	return len(classes)

def read_data(filename, resolution, d=None, normalize=True, autocontrast=True, return_image_properties=False):
	'''This method takes the path to a .txt file containing the paths to the csv files, which save information about the images. For a more detailed description of those files cf. method create_image_list(). In addition, this method takes the 'resolution' to which the images are to be scaled, a parameter 'd' which determines how many images are to be processed, a boolean parameter 'normalize' which controls whether the images shall be normalized to the interval [0,1] and a boolean parameter 'autocontrast' which controls whether a PIL intern method shall be used to increase the contrast of the images. The boolean parameter 'return_image_properties' can be set to 'True' in order to return the image list created by the create_image_list method called within this method.'''
	# create image list
	image_list = create_image_list(filename)

	# count classes
	num_classes = count_classes(image_list)

	# check whether there is a limit for the images to be loaded
	num_images = d if d is not None else len(image_list)

	# create empty arrays with appropriate size
	X = np.empty((num_images, 3, resolution[0], resolution[1]), dtype=float)
	y = np.empty((num_images))

	# iterate over images
	for idx, image in enumerate(image_list):
		if idx >= num_images:
			break

		# open the image
		im = pil.open(image['path'])

		# crop image
		im = im.crop(image['corners'])

		# autocontrast
		if autocontrast:
			im = pilops.autocontrast(im)

		# resize image to desired size
		im = im.resize(resolution)

		# save image as array within the result array
		X[idx] = np.transpose(np.asarray(im), [2,0,1])

		# save label
		y[idx] = image['label']

	# normalize images to range [0,1] if desired
	if normalize:
		X /= 255.0

	if return_image_properties:
		return X, y, num_classes, image_list
	else:
		return X, y, num_classes

if __name__ == "__main__":
	size = (48,48)
	csv_filename = "data/csv_list_train.txt"

	start_time = time.time()
	X_train, y_train = read_data(csv_filename, size, 1000, True, True)
	end_time = time.time()

	print("Execution time: " + str(end_time - start_time) + "s")
