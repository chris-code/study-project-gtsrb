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

# Stage 1: Evaluate all weights in $weights_path, and write the output to
# $results_path, keeping the folder structure.

weights=$(find $weights_path -iname "*.w" | sort) # The list of weights to evaluate
for w in $weights ; do
	generate_paths $w

	# If output already exists, don't re-compute it
	if [ -f $out_path ] ; then
		echo "Results for $w exist, skipping"
		continue
	else
		echo "Evaluating $w"
	fi

	# Create output folders if they don't exist
	out_dir="$(dirname $out_path)"
	mkdir -p "$out_dir"
	nice -n 19 python src/test.py $w $layout $dataset > $out_path

	if [ "$?" -ne "0" ] ; then
		echo "Test failed, deleting $out_path"
		rm "$out_path"
	fi
done

# Stage 2: Agglomerate the resulting values in a single file per configuration
# that is gnuplot-readable

results=$(find $results_path -iname "*.txt" | sort) # The list of results to agglomerate

# Remove old files
for result in $results ; do
	out_file="${result%.txt}"
	out_file=$(echo "$out_file" | sed -e "s/[0-9]*$//")
	out_file="${out_file}result.dat"
	rm -f "$out_file"
	#echo "# epoch accuracy" > "$out_file"
	echo "# epoch accuracy" > "$out_file"
done

# Create new ones with fresh values
for result in $results ; do
	# The file to store agglomerated results in
	out_file="${result%.txt}"
	out_file=$(echo "$out_file" | sed -e "s/[0-9]*$//")
	out_file="${out_file}result.dat"

	# Epoch number
	epoch=$(basename "$result")
	epoch="${epoch%.w}"
	epoch=$(echo "$epoch" | sed 's/[^0-9]//g')

	# Accuracy
	accuracy_line=$(tail -n 2 "$result" | head -n 1)
	accuracy=${accuracy_line##"('Test accuracy:', "}
	accuracy=${accuracy%%)}

	# Write to file
	line="$epoch $accuracy"
	echo "$line" >> "$out_file"
done
