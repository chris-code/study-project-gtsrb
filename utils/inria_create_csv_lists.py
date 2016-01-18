import os
import random
import numpy as np

# variable definitions
inriapath = "data/INRIA"
nb_testimg = 37

# initialize strings which contain the content of the csv file
csv_str_train = "Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId"
csv_str_test = "Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId"

# determine which images shall be used as test images
test_images = np.empty((15,nb_testimg))
for id, row in enumerate(test_images):
	test_images[id] = random.sample(range(186), nb_testimg)
test_images = test_images.astype(np.int16)

# iterate over files
for path, subdirs, files in os.walk(inriapath):
	# check whether we are in folder 'deFace' and skip it	
	if not path.endswith("deFace"):
		# iterate over file
		for idx, f in enumerate(files):
			# check whether file is an image
			if f.endswith(".jpg"):
				obj_id = int(path[-2:])
				for i in range(nb_testimg):
					if idx == test_images[int(obj_id) - 1, i]:
						csv_str_test += "\nPersonne" + str(obj_id).zfill(2) + "/" + f + ";384;288;0;0;384;288;" + str(obj_id - 1)
						break
				else:
					csv_str_train += "\nPersonne" + str(obj_id).zfill(2) + "/" + f + ";384;288;0;0;384;288;" + str(obj_id - 1)

with open("data/INRIA/INRIA_TRAIN.csv", "w") as text_file:
	text_file.write(csv_str_train)
with open("data/INRIA/INRIA_TEST.csv", "w") as text_file:
	text_file.write(csv_str_test)
