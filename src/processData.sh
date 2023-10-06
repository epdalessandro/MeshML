#!/bin/bash

dirName=$1
python3 preprocess_inputs.py
mkdir "../data$dirName"
mv ../data/aggregate_data.csv "../data$dirName"
mv ../data/norm* "../data$dirName"
mv ../data/processed* "../data$dirName"
mv ../data/no_outliers_data.csv "../data$dirName"
mv ../data/clipped_data.csv "../data$dirName"