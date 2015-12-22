# -*- coding: utf-8 -*-

import argparse

import keras.utils.np_utils as np_utils

import GTSRB_nn
import GTSRB_io

#~ Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument('path', help='Path to csv file that lists input images')
parser.add_argument('-e', '--epochs', help='Numper of epochs to train for [default 10]', type=int, default=10)
parser.add_argument('-b', '--batchsize', help='Size of the batches to be learned on [default 128]', type=int, default=128)
parser.add_argument('-r', '--resolution', help='Resample images to AxB resolution. [default \'48x48\']', default='48x48')
parser.add_argument('-d', '--datalimit', help='Maximum number of data points to read from PATH [if missing, read all]', type=int, default=None)
parser.add_argument('-l', '--load-weights', help='Load weights from specified file')
parser.add_argument('-s', '--store-weights', help='Store weights to specified file')
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
x_train, y_train, num_classes = GTSRB_io.read_data(args.path, resolution, args.datalimit)
y_train = np_utils.to_categorical(y_train, num_classes)

input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
model = GTSRB_nn.build_model(input_shape, num_classes)

#~ Load weights
if args.load_weights:
	print('Loading weights from {0}'.format(args.load_weights))
	model.load_weights(args.load_weights)

#~ Train the model
print('Training on {0} samples in batches of size {1} for {2} epochs'.format(x_train.shape[0], args.batchsize, args.epochs))
model.fit(x_train, y_train, nb_epoch=args.epochs, batch_size=args.batchsize, show_accuracy=True, verbose=args.verbose)

#~ Store weights
if args.store_weights:
	print('Storing weights to {0}'.format(args.store_weights))
	model.save_weights(args.store_weights)


