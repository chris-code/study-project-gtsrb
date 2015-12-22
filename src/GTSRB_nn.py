# -*- coding: utf-8 -*-

import theano
theano.config.openmp = True
import keras.models as models
import keras.layers.core as core_layers
import keras.layers.convolutional as conv_layers
import keras.optimizers as optimizers

def build_model(input_shape, num_classes=43):
	'''Builds a keras CNN model according to one column of the Multi-CNN architecture described in
	Multi-column deep neural network for traffic sign classification
	(Dan Cireşan ∗ , Ueli Meier, Jonathan Masci, Jürgen Schmidhuber)'''

	#~ Define layers
	l1 = conv_layers.Convolution2D(100, 7, 7, init='uniform', activation='tanh', input_shape=input_shape) # TODO activation function?
	l2 = conv_layers.MaxPooling2D(pool_size=(2, 2))

	l3 = conv_layers.Convolution2D(150, 4, 4, init='uniform', activation='tanh') # TODO activation function?
	l4 = conv_layers.MaxPooling2D(pool_size=(2, 2))

	l5 = conv_layers.Convolution2D(250, 4, 4, init='uniform', activation='tanh') # TODO activation function?
	l6 = conv_layers.MaxPooling2D(pool_size=(2, 2))

	l7 = core_layers.Flatten() # Transform convolutable to fully connectable layout.

	l8 = core_layers.Dense(300, init='uniform', activation='tanh')
	l9 = core_layers.Dense(num_classes, init='uniform', activation='softmax')

	#~ Add layers to a model
	model = models.Sequential()

	model.add(l1)
	model.add(l2)
	model.add(l3)
	model.add(l4)
	model.add(l5)
	model.add(l6)
	model.add(l7)
	model.add(l8)
	model.add(l9)

	#~ Configure optimizer and loss
	sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) # TODO parameters
	model.compile(loss='categorical_crossentropy', optimizer=sgd)

	return model