#!/bin/bash

cfg_file=$1

python train.py --cfg_file ./experiments/$cfg_file.yml
