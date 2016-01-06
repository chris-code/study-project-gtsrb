# -*- coding: utf-8 -*-

import argparse
import pickle # TODO is this neccessary?

import theano
theano.config.openmp = True
import keras.utils.np_utils as np_utils

import GTSRB_nn
import GTSRB_io
import GTSRB_distortions

#~ Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('path', help='Path to csv file that lists input images')
parser.add_argument('-e', '--epochs', help='Numper of epochs to train for [default 1]', type=int, default=1)
parser.add_argument('-b', '--batchsize', help='Size of the batches to be learned on [default 16]', type=int, default=16)
parser.add_argument('-r', '--resolution', help='Resample images to AxB resolution. [default \'48x48\']', default='48x48')
parser.add_argument('-d', '--datalimit', help='Maximum number of data points to read from PATH [if missing, read all]', type=int, default=None)
parser.add_argument('-m', '--morph', help='Morph training data between epochs', action='store_true')
parser.add_argument('-g', '--gray-scale', help='Determine whether the images shall be transformed to gray scale', action='store_true')
parser.add_argument('-l', '--load-status', help='Basename of the files to load status from')
parser.add_argument('-s', '--store-status', help='Basename of the files to store status in')
parser.add_argument('-v', '--verbose', help='Set the verbosity level of keras (valid values: 0, 1, 2)', type=int, default=1)
args = parser.parse_args()
try:
	sizes = args.resolution.split('x', 1)
	resolution = (int(sizes[0]), int(sizes[1]))
except:
	print('Invalid resolution specification: \'{0}\''.format(args.resolution))
	print('Resolution must be of the format \'AxB\'')
	exit(1)

#~ Load data
x_train, y_train, num_classes = GTSRB_io.read_data(args.path, resolution, args.datalimit, gray_scale=args.gray_scale)
y_train = np_utils.to_categorical(y_train, num_classes)

input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
model, optimizer = GTSRB_nn.build_model(input_shape, num_classes)

#~ Load status
if args.load_status:
	print('Loading status from {0}'.format(args.load_status))

	#~ weights
	weight_filename = args.load_status + ".w"
	model.load_weights(weight_filename)

	#~ training parameters
	train_filename = args.load_status + ".t"
	with open(train_filename, 'rb') as train_file:
		optimizer_state = pickle.load(train_file)
	optimizer.set_state(optimizer_state)

#~ Create distortions callback
if args.morph:
	distcall = GTSRB_distortions.Distortions(x_train, x_train.shape[0])
	callbacks = [distcall]
	print('Distortions will be applied to training data between epochs')
else:
	callbacks = []

#~ Train the model
print('Training on {0} samples in batches of size {1} for {2} epochs'.format(x_train.shape[0], args.batchsize, args.epochs))
model.fit(x_train, y_train, nb_epoch=args.epochs, callbacks=callbacks, batch_size=args.batchsize, show_accuracy=True, verbose=args.verbose)

#~ Store status
if args.store_status:
	print('Storing status to {0}'.format(args.store_status))

	#~ weights
	weight_filename = args.store_status + ".w"
	model.save_weights(weight_filename, overwrite=True)

	#~ training parameters
	train_filename = args.store_status + ".t"
	optimizer_state = optimizer.get_state()
	with open(train_filename, 'wb') as train_file:
		pickle.dump(optimizer_state, train_file, pickle.HIGHEST_PROTOCOL)

print('Done')
