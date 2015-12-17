import argparse

import numpy as np
import PIL.Image as pil

import GTSRB_nn

def visualize_filters(model, path, verbose):
	# add "/" to path, if it does not end on this character
	if path[-1] != "/":
		path += "/"

	# iterate over layers
	for layer_id,layer in enumerate(model.layers):
		# print layer id
		if verbose:
			print("Layer " + str(layer_id) + ":")

		# ignore all layers, which are no convolutional layers
		if layer.get_config()['name'] == "Convolution2D":
			# iterate over filters
			for weight_matrices in layer.get_weights():
				# print filter shape
				if verbose:
					print(weight_matrices.shape)
			
				# ignore 1 dimensional filters
				if len(weight_matrices.shape) > 1:		
					num_outputs, num_filters, xdim, ydim = weight_matrices.shape
					if verbose:
						print("number of outputs:" + str(num_outputs))
						print("number of filters:" + str(num_filters))
						print("filter size: " + str(xdim) + "x" + str(ydim))

					width = num_filters * (xdim + 1) + 1
					height = num_outputs * (ydim + 1) + 1
					filter_collection = pil.new("L", (width, height), "white")
				
					for i_id, inputs in enumerate(weight_matrices):
						for f_id, filter in enumerate(inputs):
							xpos = f_id * (xdim+1) + 1
							ypos = i_id * (ydim+1) + 1
							minval = np.min(filter)
							filter += minval
							maxval = np.max(filter)
							filter *= 255./maxval
							im_filter = pil.fromarray(filter)

							filter_collection.paste(im_filter, (xpos, ypos))

					filter_collection.save(path + "weights_on_layer_" + str(layer_id) + ".png")


#~ Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument('loadpath', help='Path to the weights which are to be loaded')
parser.add_argument('-s', '--savepath', help='Path to save location of the visualized filters')
parser.add_argument('-v', '--verbose', help='Determine whether the programm shall print information in the terminal or not', action="store_true")
args = parser.parse_args()

# build model
input_shape = (3, 48, 48)
model = GTSRB_nn.build_model(input_shape)

#~ Load weights
if args.verbose:
	print('Loading weights from \"{0}\"'.format(args.loadpath))
model.load_weights(args.loadpath)

# print number of layers
if args.verbose:
	print("Number of layers: " + str(len(model.layers)))

# visualize filters
visualize_filters(model, args.savepath, args.verbose)
