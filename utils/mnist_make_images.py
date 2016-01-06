import sklearn.datasets
import PIL.Image as pil
import numpy as np
import math
import os

# create folders
def create_folders():
	os.makedirs("data/MNIST_TRAIN")
	os.makedirs("data/MNIST_TEST")
	for i in range(10):
		name = str(i).zfill(5)
		os.makedirs("data/MNIST_TRAIN/" + name)

create_folders()

# load MNIST and prepare some counters
mnist = sklearn.datasets.fetch_mldata('MNIST original')
previouslabel = 0
localcounter = 0

# save training images
for globalcounter, image in enumerate(mnist.data[:60000]):
	print(globalcounter)

	# determine label
	label = int(mnist.target[globalcounter])	
	labelstr = str(int(mnist.target[globalcounter])).zfill(5)

	if previouslabel != label:
		previouslabel = label
		localcounter = 0

	# save image
	im = pil.fromarray(np.asarray(image).reshape(28,28))
	im.save("data/MNIST_TRAIN/" + labelstr + "/" + str(localcounter).zfill(5) + ".ppm")

	localcounter += 1

# create csv files
csv_file_list_str = ""
csv_global_counter = 0
for cls in range(10):
	csv_local_counter = 0
	csv_file_str = "Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId"
	while int(mnist.target[csv_global_counter]) == cls:
		csv_file_str += "\n" + str(csv_local_counter).zfill(5) + ".ppm;28;28;0;0;28;28;" + str(int(mnist.target[csv_global_counter]))
		csv_local_counter += 1
		csv_global_counter += 1
	with open("data/MNIST_TRAIN/" + str(cls).zfill(5) + "/MNIST-" + str(cls).zfill(5) + ".csv", "w") as text_file:
		text_file.write(csv_file_str)
		csv_file_list_str += "data/MNIST_TRAIN/" + str(cls).zfill(5) + "/MNIST-" + str(cls).zfill(5) + ".csv\n"
csv_file_list_str = csv_file_list_str[:-1]
with open("data/csv_list_train_mnist.txt", "w") as csv_list_file:
	csv_list_file.write(csv_file_list_str)

# save test images
csv_file_str = "Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId"
global_counter = 60000
local_counter = 0

for image in mnist.data[60000:]:
	print(global_counter)

	# save image
	label = int(mnist.target[global_counter])	
	labelstr = str(int(mnist.target[global_counter])).zfill(5)
	im = pil.fromarray(np.asarray(image).reshape(28,28))
	im.save("data/MNIST_TEST/" + str(local_counter).zfill(5) + ".ppm")
	csv_file_str += "\n" + str(local_counter).zfill(5) + ".ppm;28;28;0;0;28;28;" + str(int(mnist.target[global_counter]))

	global_counter += 1
	local_counter += 1


with open("data/MNIST_TEST/MNIST-TEST.csv", "w") as text_file:
	text_file.write(csv_file_str)
with open("data/csv_list_test_mnist.txt", "w") as csv_list_file:
	csv_list_file.write("data/MNIST_TEST/MNIST-TEST.csv")
