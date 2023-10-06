#!/bin/bash

FOLDER=$1
mkdir ../results/$FOLDER
nohup python3 training.py ../data/$FOLDER/aggregate_data.csv > ../results/$FOLDER/aggregate.txt &
nohup python3 training.py ../data/$FOLDER/clipped_data.csv > ../results/$FOLDER/clipped.txt &
nohup python3 training.py ../data/$FOLDER/no_outliers_data.csv > ../results/$FOLDER/no_outliers.txt &
nohup python3 training.py ../data/$FOLDER/norm_clipped_data.csv > ../results/$FOLDER/norm_clipped.txt &
nohup python3 training.py ../data/$FOLDER/processed_clipped_data.csv > ../results/$FOLDER/processed_clipped.txt &
nohup python3 training.py ../data/$FOLDER/processed_outlier_data.csv > ../results/$FOLDER/processed_outlier.txt &