import argparse
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '../src'))
import nn

parser = argparse.ArgumentParser()
parser.add_argument('source_layout', metavar='source-layout', help='Path source network layout specification')
parser.add_argument('source_weights', metavar='source-weights', help='Path sourcce network weights')
parser.add_argument('target_layout', metavar='target-layout', help='Path target network layout specification')
parser.add_argument('target_weights', metavar='target-weights', help='Path target network weights')
parser.add_argument('layerspec', help='Which layers to copy. Format: a-b-...-z where a-z are 0-based layer numbers')
args = parser.parse_args()

#~ Load source model
print('Loading source model from {0}'.format(args.source_layout))
source_layout = nn.load_layout(args.source_layout)
source_model, source_optimizer = nn.build_model_to_layout(source_layout)

#~ Load source weights
print('\tLoading source weights from {0}'.format(args.source_weights))
source_model.load_weights(args.source_weights)

#~ Load target model
print('Loading target model from {0}'.format(args.target_layout))
target_layout = nn.load_layout(args.target_layout)
target_model, target_optimizer = nn.build_model_to_layout(target_layout)

#~ Load target weights
if os.path.isfile(args.target_weights):
	print('\tLoading target weights from {0}'.format(args.target_weights))
	target_model.load_weights(args.target_weights)
else:
	print('\tNot loading target weights. Initialization values will be used where not overwritten.')

layers = args.layerspec.split('-')
layers = [int(l) for l in layers]

#~ Copy layer weights
for layer_idx in layers:
	print('Copying weights on layer {0}'.format(layer_idx))
	weights = source_model.layers[layer_idx].get_weights()
	target_model.layers[layer_idx].set_weights(weights)

#~ Store target weights
print('Storing target weights to {0}'.format(args.target_weights))
target_model.save_weights(args.target_weights, overwrite=True)












