#!/bin/bash

# To use this, set ADDITIONAL_LIBS to the folder containing the required
# libaries and source this file.

# This script will append to the LD_LIBRARY_PATH environment variable so that
# the libraries located in ADDITIONAL_LIBS will be found by ld when running
# the program. Libraries that need to be found in ADDITIONAL_LIBS if they are
# not found in the regular system paths include:
# libopenblas.so 
# libopenblas.so.0
# libhdf5_serial.so.10
# libhdf5_serial_hl.so.10
# libhdf5_serial_hl.so.10.0.1

ADDITIONAL_LIBS=/path/to/libraries

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ADDITIONAL_LIBS
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
