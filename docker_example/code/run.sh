#!/bin/bash

# $1 is the directory of the test dataset images './inputs/{int_video}/'
# $2 is the filepath of the annotation predictions './outputs/{int_video}.csv'

python main.py --pt_input=$1 --pt_output=$2
