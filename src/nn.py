# -*- coding: utf-8 -*-

import argparse
import json

import theano
theano.config.openmp = True
import keras.models as models
import keras.layers.core as core_layers
import keras.layers.convolutional as conv_layers
import keras.optimizers as optimizers

def build_model_to_layout(layout, momentum=0.0, nesterov=False):
	model = models.Sequential()

	for ltype, lspec in layout:
		if ltype == 'conv2D':
			layer = conv_layers.Convolution2D(**lspec)
		elif ltype == 'maxpool2D':
			layer = conv_layers.MaxPooling2D(**lspec)
		elif ltype == 'flatten':
			layer = core_layers.Flatten()
		elif ltype == 'dense':
			layer = core_layers.Dense(**lspec)
		else:
			raise NotImplementedError

		model.add(layer)

	sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=momentum, nesterov=nesterov) # TODO parameters
	model.compile(loss='categorical_crossentropy', optimizer=sgd)

	return model, sgd

def get_gtsrb_layout(input_shape, num_classes):
	'''Builds a keras CNN layout according to one column of the Multi-CNN architecture described in
	Multi-column deep neural network for traffic sign classification
	(Dan Cireşan ∗ , Ueli Meier, Jonathan Masci, Jürgen Schmidhuber)'''

	layout = []
	layout.append( ('conv2D', {'nb_filter': 100, 'nb_row': 7, 'nb_col': 7, 'init': 'uniform', 'activation': 'tanh', 'input_shape': input_shape}) )
	layout.append( ('maxpool2D', {'pool_size': (2,2)}) )
	layout.append( ('conv2D', {'nb_filter': 150, 'nb_row': 4, 'nb_col': 4, 'init': 'uniform', 'activation': 'tanh'}) )
	layout.append( ('maxpool2D', {'pool_size': (2,2)}) )
	layout.append( ('conv2D', {'nb_filter': 250, 'nb_row': 4, 'nb_col': 4, 'init': 'uniform', 'activation': 'tanh'}) )
	layout.append( ('maxpool2D', {'pool_size': (2,2)}) )
	layout.append( ('flatten', None) )
	layout.append( ('dense', {'output_dim': 300, 'init': 'uniform', 'activation': 'tanh'}) )
	layout.append( ('dense', {'output_dim': num_classes, 'init': 'uniform', 'activation': 'softmax'}) )

	return layout

def get_gtsrb_fixed_filter_layout(input_shape, num_classes):
	pass

def get_mnist_layout(input_shape, num_classes):
	layout = []
	layout.append( ('conv2D', {'nb_filter': 20, 'nb_row': 5, 'nb_col': 5, 'init': 'uniform', 'input_shape': input_shape}) )
	layout.append( ('maxpool2D', {'pool_size': (2,2)}) )
	layout.append( ('conv2D', {'nb_filter': 20, 'nb_row': 5, 'nb_col': 5, 'init': 'uniform'}) )
	layout.append( ('maxpool2D', {'pool_size': (2,2)}) )
	layout.append( ('flatten', None) )
	layout.append( ('dense', {'output_dim': 500, 'init': 'uniform', 'activation': 'relu'}) )
	layout.append( ('dense', {'output_dim': num_classes, 'init': 'uniform', 'activation': 'softmax'}) )

	return layout

def load_layout(path):
	with open(path) as in_file:
		layout = json.load(in_file)
	return layout

def store_layout(layout, path):
	with open(path, 'w') as out_file:
		json.dump(layout, out_file, indent=2, sort_keys=True)

if __name__ == '__main__':
	#~ Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('path', help='Folder to store known layouts in')
	args = parser.parse_args()

	if args.path[-1] == '/':
		gtsrb_path = args.path + 'gtsrb.l'
		mnist_path = args.path + 'mnist.l'
	else:
		gtsrb_path = args.path + '/gtsrb.l'
		mnist_path = args.path + '/mnist.l'
	store_layout(get_gtsrb_layout((3, 48, 48), 43), gtsrb_path)
	store_layout(get_mnist_layout((1, 48, 48), 10), mnist_path)

	l = load_layout(gtsrb_path)
	build_model_to_layout(l)