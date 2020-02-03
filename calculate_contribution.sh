#!/bin/bash

cfg_file=$1

python calculate_contribution.py --cfg_file ./experiments/$cfg_file.yml
