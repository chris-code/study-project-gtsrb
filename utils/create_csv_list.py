import os

def search_folder(path):
	list = ""
	for subdir, dirs, files in os.walk(path):
		for file in files:
			if file[-4:] == ".csv":
				list += subdir + "/" + file + "\n"
	return list

path = "data/GTSRB_TEST/"

list = search_folder(path)[:-1]

with open("csv_list_test.txt", "w") as list_file:
	list_file.write(list)
