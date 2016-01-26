# -*- coding: utf-8 -*-

import argparse
import sys
import os

import numpy as np
import PIL.Image as pil_img

sys.path.insert(1, os.path.join(sys.path[0], '../src'))
import nn

def visualize_filters(model, path):
	# add "/" to path, if it does not end on this character
	if path[-1] != "/":
		path += "/"

	# iterate over layers
	for layer_id,layer in enumerate(model.layers):
		# ignore all layers which are not convolutional layers
		if layer.get_config()['name'] != "Convolution2D":
			continue

		# iterate over outputs
		for weight_matrices in layer.get_weights():
			# ignore 1-dimensional filters
			if len(weight_matrices.shape) > 1:
				# get number of output maps, number of filters per input map and the x and y dimensions of the filters
				num_outputs, num_filters, xdim, ydim = weight_matrices.shape

				# calculate the size of the filter collection image and create it
				width = num_filters * (xdim + 1) + 1
				height = num_outputs * (ydim + 1) + 1
				filter_collection = pil_img.new("L", (width, height), "white")

				# iterate over the output maps of the next layer
				for i_id, inputs in enumerate(weight_matrices):
					# iterate over the filters of the maps on the current layer
					for f_id, filter in enumerate(inputs):
						# determine position of the filter in the filter collection
						xpos = f_id * (xdim+1) + 1
						ypos = i_id * (ydim+1) + 1

						# rescale the grayvalues to visualize them
						filter -= np.min(filter)
						filter *= 255./np.max(filter)
						im_filter = pil_img.fromarray(filter)

						# paste the filter into the filter collection
						filter_collection.paste(im_filter, (xpos, ypos))

				# save the filter collection of layer 'id'
				filter_collection.save(path + "weights_on_layer_" + str(layer_id) + ".png")

if __name__ == "__main__":
	#~ Parse parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('weights', help='Path to the weights which are to be loaded')
	parser.add_argument('layout', help='Path network layout specification')
	parser.add_argument('-s', '--savepath', help='Path to save location of the visualized filters', default='./')
	parser.add_argument('-v', '--verbose', help='Determine whether the programm shall print information in the terminal or not', action="store_true")
	args = parser.parse_args()

	# Load model
	print('Loading model from {0}'.format(args.layout))
	layout = nn.load_layout(args.layout)
	model, optimizer = nn.build_model_to_layout(layout)

	#~ Load weights
	print('Loading weights from \"{0}\"'.format(args.weights))
	model.load_weights(args.weights)

	# visualize filters
	print('Generating visualizations and storing to {0}'.format(args.savepath))
	visualize_filters(model, args.savepath)

	print('Done')
