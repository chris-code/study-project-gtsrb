import argparse

import numpy as np
import keras.models as models
import keras.layers.core as core_layers
import keras.layers.convolutional as conv_layers
import keras.optimizers as optimizers
import keras.utils.np_utils as np_utils

import GTSRB_io

def build_model(input_shape):
	model = models.Sequential()

	l1 = conv_layers.Convolution2D(100, 7, 7, init='uniform', activation='tanh', input_shape=input_shape) # TODO activation function?
	l2 = conv_layers.MaxPooling2D(pool_size=(2, 2))

	l3 = conv_layers.Convolution2D(150, 4, 4, init='uniform', activation='tanh') # TODO activation function?
	l4 = conv_layers.MaxPooling2D(pool_size=(2, 2))

	l5 = conv_layers.Convolution2D(250, 4, 4, init='uniform', activation='tanh') # TODO activation function?
	l6 = conv_layers.MaxPooling2D(pool_size=(2, 2))

	l7 = core_layers.Dense(300, init='uniform', activation='tanh')
	l8 = core_layers.Dense(43, init='uniform', activation='softmax')

	model.add(l1)
	model.add(l2)
	model.add(l3)
	model.add(l4)
	model.add(l5)
	model.add(l6)
	model.add(core_layers.Flatten())
	model.add(l7)
	model.add(l8)

	sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) # TODO parameters
	model.compile(loss='categorical_crossentropy', optimizer=sgd)

	return model

parser = argparse.ArgumentParser()
parser.add_argument('path', help='Path to csv file that lists input images')
parser.add_argument('-e', '--epochs', help='Numper of epochs to train for', type=int, default=10)
parser.add_argument('-r', '--resolution', help='Resample images to AxB resolution. Default is 48x48.', default='48x48')
parser.add_argument('-d', '--datalimit', help='Maximum number of data points to read from PATH', type=int, default=None)
parser.add_argument('-l', '--load-weights', help='Load weights from specified file')
parser.add_argument('-s', '--store-weights', help='Store weights to specified file')
parser.add_argument('-v', '--verbose', help='Set the verbosity level of keras (valid values: 0, 1, 2)', type=int, default=1)
args = parser.parse_args()
try:
	sizes = args.resolution.split('x', maxsplit=1)
	resolution = (int(sizes[0]), int(sizes[1]))
except:
	print('Invalid resolution specification: \'{0}\''.format(args.resolution))
	print('Resolution must be of the format \'AxB\'')
	exit(1)

x_train, y_train = GTSRB_io.read_data(args.path, resolution, args.datalimit)
class_count = int(np.max(y_train) + 1)
if class_count != 43:
	print('There are {0} classes instead of 43!'.format(class_count))
	exit()
y_train = np_utils.to_categorical(y_train, class_count)

input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
model = build_model(input_shape)
#~ model.summary()

if args.load_weights:
	print('Loading weights from {0}'.format(args.load_weights))
	model.load_weights(args.load_weights)

model.fit(x_train, y_train, nb_epoch=args.epochs, show_accuracy=True, verbose=args.verbose)

if args.store_weights:
	print('Storing weights to {0}'.format(args.store_weights))
	model.save_weights(args.store_weights)


