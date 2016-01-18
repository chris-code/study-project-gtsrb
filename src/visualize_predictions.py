# -*- coding: utf-8 -*-

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
	for i, c, p in zip(image_info, class_ids, probabilites):
		fig, axes = ppt.subplots(1, c.size + 1, figsize=(12, 6), subplot_kw={'xticks': [], 'yticks': []})
		image = image_loader.get_image(i['path'])
		axes[0].imshow(image, interpolation='none')
		axes[0].set_title('Input', y=-0.15)

		for idx, (class_id, probability) in enumerate(zip(c,p)):
			class_representative = image_loader.get_class_representative(class_id)
			axes[idx+1].imshow(class_representative, interpolation='none')
			axes[idx+1].set_title('{0:.4f}'.format(probability), y=-0.15)

		ppt.show()

def get_candidates(probabilities, num_candidates=3):
	num_samples = probabilities.shape[0]

	class_ids = np.argsort(probabilities, axis=1)
	class_ids = class_ids[:,-num_candidates:] # Take the 3 most likely classes
	class_ids = class_ids[:,::-1] # Put them in descending order

	# Also generate probabilities of the num_candidates most likely classes
	row_indices = np.arange(num_samples).reshape((num_samples, 1))
	winner_probabilities = probabilities[row_indices, class_ids]

	return class_ids, winner_probabilities

if __name__ == '__main__':
	#~ Parse parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('path', help='Location of the stored predictions')
	parser.add_argument('-c', '--candidates', help='Number of candidates to show for each test image [default 3]', type=int, default=3)
	parser.add_argument('-d', '--datalimit', help='Maximum number of data points to read from DATA [if missing, read all]', type=int, default=None)
	args = parser.parse_args()

	with open(args.path, 'rb') as in_file:
		image_info, probabilities = pickle.load(in_file)

	if args.datalimit:
		image_info = image_info[:args.datalimit]
		probabilities = probabilities[:args.datalimit,...]

	class_ids, winner_probabilities = get_candidates(probabilities, num_candidates=args.candidates)

	image_loader = Image_Loader((48, 48), 'visualizations/representatives/class{0}.ppm')

	visualize(image_info, class_ids, winner_probabilities, image_loader)
