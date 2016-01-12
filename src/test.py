# -*- coding: utf-8 -*-

import argparse

import theano
theano.config.openmp = True
import keras.utils.np_utils as np_utils

import nn
import dataset_io

#~ Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument('weights', help='Path weights in xyz.w file')
parser.add_argument('path', help='Path to csv file that lists input images')
parser.add_argument('-b', '--batchsize', help='Size of the batches to be learned on [default 16]', type=int, default=16)
parser.add_argument('-r', '--resolution', help='Resample images to AxB resolution. [default \'48x48\']', default='48x48')
parser.add_argument('-d', '--datalimit', help='Maximum number of data points to read from PATH [if missing, read all]', type=int, default=None)
parser.add_argument('-v', '--verbosity', help='Set the verbosity level of keras (valid values: 0, 1, 2)', type=int, default=1)
args = parser.parse_args()
try:
	sizes = args.resolution.split('x', 1)
	resolution = (int(sizes[0]), int(sizes[1]))
except:
	print('Invalid resolution specification: \'{0}\''.format(args.resolution))
	print('Resolution must be of the format \'AxB\'')
	exit(1)

#~ Load data
x_test, y_test, num_classes = dataset_io.read_data(args.path, resolution, args.datalimit)
y_test = np_utils.to_categorical(y_test, num_classes)

input_shape = (x_test.shape[1], x_test.shape[2], x_test.shape[3])
layout = nn.get_gtsrb_layout(input_shape, num_classes)
model, optimizer = nn.build_model_to_layout(layout)

#~ Load weights
print('Loading weights from {0}'.format(args.weights))
model.load_weights(args.weights)

#~ Test the model
print('Testing on {0} samples at resolution {1}x{2} in batches of size {3}'.format(x_test.shape[0], resolution[0], resolution[1], args.batchsize))
score = model.evaluate(x_test, y_test, batch_size=args.batchsize, show_accuracy=True, verbose=args.verbosity)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print('Done')