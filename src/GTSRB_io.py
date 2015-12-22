import PIL.Image as pil
import numpy as np
import time
import random
import csv

def create_image_list(filename):
	'''This method takes the path to a list of csv files, which contain information about the images.'''
	# open file which contains the csv list
	with open(filename, "r") as f:
		# read csv paths from file and save them in a list
		csv_paths = f.readlines()
		
		# create list, which is supposed to save information about the images
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

def read_data(filename, resolution, d=None, normalize=True):
	'''This method takes a file containing the csv paths and returns a 4D array containing the image data and a 1D array containing the labels.'''
	# create image list
	image_list = create_image_list(filename)

	# count classes
	classes = set()
	for image in image_list:
		classes.add(image['label'])
	num_classes = len(classes)

	# check whether there is a limit for the images to be loaded
	number_of_images = d if d is not None else len(image_list)

	# create empty arrays with appropriate size
	X = np.empty((number_of_images, 3, resolution[0], resolution[1]), dtype=float)
	y = np.empty((number_of_images))

	# iterate over images
	for idx, image in enumerate(image_list):
		if idx >= number_of_images:
			break;
		
		# open the image
		im = pil.open(image['path'])

		# crop image
		im = im.crop(image['corners'])

		# resize image to desired size
		im = im.resize(resolution)
		
		# save image as array within the result array
		X[idx] = np.transpose(np.asarray(im), [2,0,1])

		# save label
		y[idx] = image['label']

	if normalize:
		X[idx] /= 255

	return X, y, num_classes

if __name__ == "__main__":
	size = (48,48)
	csv_filename = "data/csv_list_train.txt"

	start_time = time.time()
	X_train, y_train = read_data(csv_filename, size, 1000, true)
	end_time = time.time()

	print("Execution time: " + str(end_time - start_time) + "s")
