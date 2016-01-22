import sys
import os

for i in range(26,31):
	os.system("python src/train.py layouts/inria_no_filter_train.l data/csv_list_train_inria.txt -m -l weights/inria/gtsrb_filters_" + str(i - 1) + " -s weights/inria/gtsrb_filters_" + str(i))
