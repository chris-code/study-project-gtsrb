import os
import random
import numpy as np

# variable definitions
coilpath = "data/COIL100"
nb_testimg = 14

# initialize strings which contain the content of the csv file
csv_str_train = "Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId"
csv_str_test = "Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId"

# determine which images shall be used as test images
test_images = np.empty((100,nb_testimg))
for id, row in enumerate(test_images):
	test_images[id] = random.sample(range(72), nb_testimg)
test_images = test_images.astype(np.int16) * 5

# iterate over files
for subdir, dirs, files in os.walk(coilpath):
	for f in files:
		# check whether file is an image
		if f[-4:] == ".png":
			# extract object id and degree from file name
			obj_id, degree = f[3:-4].split('__', 1)
			# append image to respective csv file
			for i in range(nb_testimg):
				if int(degree) == test_images[int(obj_id) - 1, i]:
					csv_str_test += "\nobj" + obj_id + "__" + degree + ".png;128;128;0;0;128;128;" + str(int(obj_id) - 1)
					break
			else:
				csv_str_train += "\nobj" + obj_id + "__" + degree + ".png;128;128;0;0;128;128;" + str(int(obj_id) - 1)

with open("data/COIL100/COIL_TRAIN.csv", "w") as text_file:
	text_file.write(csv_str_train)
with open("data/COIL100/COIL_TEST.csv", "w") as text_file:
	text_file.write(csv_str_test)
