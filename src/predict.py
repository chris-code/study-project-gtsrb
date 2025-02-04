# -*- coding: utf-8 -*-

import argparse
import pickle

import numpy as np
import keras.utils.np_utils as np_utils

import dataset_io
import nn

def extract_misclassified(image_properties, predictions, labels):
	'''Given a list of image properties and the class probabilities, extract the image properties and
	class probabilities of the misclassified samples.

	image_properties: list type, content can be anything
	predictions: numpy array of shape (samples, classes)
	labels: numpy array of shape (samples, classes), labels in one-hot encoding

	returns wrong_image_properties: the part of image_properties that belongs to misclassified images
	returns wrong_predictions: the part of predictions that belongs to misclassified images'''

	pred_labels = np.argmax(predictions, axis=1) # from one-hot encoding to a class id
	labels = np.argmax(labels, axis=1) # same

	mistakes = pred_labels != labels # check which images are misclassified
	wrong_image_properties = [prop for prop, m in zip(image_properties, mistakes) if m]
	wrong_predictions = predictions[mistakes]

	return wrong_image_properties, wrong_predictions

#~ Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument('weights', help='Path weights in xyz.w file')
parser.add_argument('layout', help='Path network layout specification')
parser.add_argument('data', help='Path to csv file that lists input images')
parser.add_argument('path', help='Location for storing the predictions')
parser.add_argument('-b', '--batchsize', help='Size of the batches to be learned on [default 16]', type=int, default=16)
parser.add_argument('-d', '--datalimit', help='Maximum number of data points to read from DATA [if missing, read all]', type=int, default=None)
parser.add_argument('-m', '--misclassified', help='Only store predictions for missclassified images', action='store_true')
args = parser.parse_args()

#~ Load model
print('Loading model from {0}'.format(args.layout))
layout = nn.load_layout(args.layout)
model, optimizer = nn.build_model_to_layout(layout)

#~ Load weights
print('Loading weights from {0}'.format(args.weights))
model.load_weights(args.weights)

#~ Load data
input_shape = layout[0][1]['input_shape']
resolution = input_shape[1:]
print('Loading data from {0} and rescaling it to {1}x{2}'.format(args.data, resolution[0], resolution[1]))
x_test, y_test, num_classes, image_properties = dataset_io.read_data(args.data, resolution, args.datalimit, return_image_properties=True)
y_test = np_utils.to_categorical(y_test, num_classes)

#~ Create predictions
print('Predicting labels for {0} samples at resolution {1}x{2} in batches of size {3}'.format(x_test.shape[0], resolution[0], resolution[1], args.batchsize))
predictions = model.predict_proba(x_test, batch_size=args.batchsize)

#~ If desired, reduce output to the misclassified images
if args.misclassified:
	print('Reducing output to misclassified samples')
	sample_count = len(image_properties)
	image_properties, predictions = extract_misclassified(image_properties, predictions, y_test)
	print('\tMisclassified images: {0} out of {1}'.format(len(image_properties), sample_count))

#~ Store preditions on disk
print('Storing samples to {0}'.format(args.path))
with open(args.path, 'wb') as out_file:
	pickle.dump((image_properties, predictions), out_file, pickle.HIGHEST_PROTOCOL)

print('Done')
