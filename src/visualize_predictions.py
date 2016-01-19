# -*- coding: utf-8 -*-

'''If executed, this program will display images, together with the specified number of most likely
candidates and their probabilities. For the candidates, class representatives are used that reside
inside a specified class representative folder and are named classCLASSID.ppm'''

import argparse
import pickle

import PIL.Image as pil_img
import PIL.ImageOps as pil_img_ops
import numpy as np
import matplotlib.pyplot as ppt

class Image_Loader():
	'''Provides methods to load an image from a given path and to load an image according to a class
	id, from a folder containing class representatives, which has to be specified beforehand'''

	def __init__(self, resolution, class_representative_path):
		'''resolution: 2-tuple of x- and y-resolution
		class_representative_path: path to a folder of class representative images named class$.ppm where $ is the class id'''
		self.resolution = resolution
		self.class_representative_path = class_representative_path

	def get_image(self, path):
		'''load an image and rescale it to the resolution specified in __init__
		path: path to the image file
		returns: the image'''
		image = pil_img.open(path)
		image = image.resize(self.resolution)
		return image

	def get_class_representative(self, class_id):
		'''load image that represents a certain class in the resolution specified in __init__
		class_id: id of the class
		returns: representative image'''
		path = self.class_representative_path.format(class_id)
		return self.get_image(path)

def visualize(image_info, class_ids, probabilites, image_loader):
	'''Displays an image, the most likely candidates and their probabilities
	image_info: iterable that contains dictionaries, where the key "path" provides the path to an image as a string
	class_ids: numpy array of shape (number of samples, number of candidates), contains candidate class ids
	probabilites: numpy array of shape (number of samples, number of candidates), contains candidate class probabilities
	image_loader: Instance of Image_Loader
	'''

	#~ Iterate over sample images with their respective candidates and their probabilities
	for i, c, p in zip(image_info, class_ids, probabilites):
		fig, axes = ppt.subplots(1, c.size + 1, figsize=(12, 6), subplot_kw={'xticks': [], 'yticks': []})

		#~ Load and display the image
		image = image_loader.get_image(i['path'])
		axes[0].imshow(image, interpolation='none')
		axes[0].set_title('Input', y=-0.15)

		#~ Load and display representatives of the candidate classes and their probabilities
		for idx, (class_id, probability) in enumerate(zip(c,p)):
			class_representative = image_loader.get_class_representative(class_id)
			axes[idx+1].imshow(class_representative, interpolation='none')
			axes[idx+1].set_title('{0:.4f}'.format(probability), y=-0.15)

		ppt.show()

def get_candidates(probabilities, num_candidates=3):
	'''Transform the class probabilites into the num_candidates most likely class ids and their probabilites
	probabilites: numpy array of shape (samples, classes)
	num_candidates: integer, the number of candidates to extract
	returns class_ids: numpy array of shape (samples, num_candidates), the ids of the most likely classes
	returns winner_probabilities: numpy array of shape (samples, num_candidates), the corresponding probabilites to class_ids'''

	num_samples = probabilities.shape[0]

	#~ Generate the ids of the most likely classes (the most likely one should be at the lowest index)
	class_ids = np.argsort(probabilities, axis=1)
	class_ids = class_ids[:,-num_candidates:] # Take the num_candidates most likely classes
	class_ids = class_ids[:,::-1] # Put them in descending order

	# Also generate probabilities of the num_candidates most likely classes
	row_indices = np.arange(num_samples).reshape((num_samples, 1))
	winner_probabilities = probabilities[row_indices, class_ids]

	return class_ids, winner_probabilities

if __name__ == '__main__':
	#~ Parse parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('predictions', help='Location of the stored predictions')
	parser.add_argument('representatives', help='Path to the directory containing class representatives')
	parser.add_argument('-c', '--candidates', help='Number of candidates to show for each test image [default 3]', type=int, default=3)
	parser.add_argument('-d', '--datalimit', help='Maximum number of data points to read from DATA [if missing, read all]', type=int, default=None)
	args = parser.parse_args()

	#~ Read data, a list of the sample image properties (including the path), and the class probabilites calculated for them
	#~ Data should be a pickle dump of (image_info, probabilities) akin to the one produced by predict.py
	#~ image_info is a list of {'path': path_as_string} dictionaries (may contain other key-value pairs)
	#~ probabilities is a (number_of_samples, number_of_classes) numpy array.
	with open(args.predictions, 'rb') as in_file:
		image_info, probabilities = pickle.load(in_file)

	#~ Limit the amount of data if specified per parameter
	if args.datalimit:
		image_info = image_info[:args.datalimit]
		probabilities = probabilities[:args.datalimit,...]

	#~ Get class ids and their probabilities of the most likely classes
	class_ids, winner_probabilities = get_candidates(probabilities, num_candidates=args.candidates)

	#~ Actual visualization
	if args.representatives[-1] != '/':
		args.representatives += '/'
	class_representative_path_spec = args.representatives + 'class{0}.ppm'
	image_loader = Image_Loader((48, 48), class_representative_path_spec)
	visualize(image_info, class_ids, winner_probabilities, image_loader)
