#!/bin/bash

# Path settings
weights_path="weights/"
results_path="results/"

# Will get path to layout, dataset and out_path for a weight path in $1.
# They are stored in layout, dataset and out_path, respectively
generate_paths() {
	layout_hint="${1%.w}"
	layout_hint=$(echo "$layout_hint" | sed -e "s/[0-9]*$//")
	layout_hint="${layout_hint}layout.txt"
	layout=$(cat "$layout_hint")

	dataset_hint=$(dirname $1)
	dataset_hint="${dataset_hint}/test_dataset.txt"
	dataset=$(cat "$dataset_hint")

	out_path="${1##$weights_path}"
	out_path="${out_path%.w}"
	out_path="${results_path}${out_path}.txt"
}

weights=$(find $weights_path -iname *.w | sort) # The list of weights to evaluate
for w in $weights ; do
	echo "Evaluating $w"

	generate_paths $w

	# If output already exists, don't re-compute it
	if [ -f $out_path ] ; then
		echo "Results for $w exist, skipping"
		continue
	fi

	# Create output folders if they don't exist
	out_dir="$(dirname $out_path)"
	mkdir -p "$out_dir"
	nice -n 19 python src/test.py $w $layout $dataset > $out_path
done
