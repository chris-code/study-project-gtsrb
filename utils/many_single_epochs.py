import sys
import os

for i in range(20,31):
	os.system("python src/train.py layouts/inria_no_filter_train.l data/csv_list_train_inria.txt -m -l weights/inria/gtsrb_filters_0" + str(i - 1) + " -s weights/inria/gtsrb_filters_0" + str(i))
