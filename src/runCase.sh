#!/bin/bash

# Run from mesh_generation
NUM_RUNS=100
LOCATION="../jobs/xfa"
FILE_NAME="training_data"

for ((run = 1; run <= $NUM_RUNS; run++)); do
    if test -f $LOCATION/run-$run.xfa; then
        ../../bin/xf_MeshStats -in "$LOCATION/run-$run.xfa" -wallDistance True -gWallDistance True
        ../../bin/xf_Refine -in "$LOCATION/run-$run-unref.gri" -interact True < ../input.txt
        python3 process_data.py Q1_refined.gri WallDistance.txt GradientWallDistance.txt "$FILE_NAME$run.csv" $run "../jobs/scripts/param.txt"
        mv "$FILE_NAME$run.csv" ../data/training/
    else
        echo "$LOCATION/run-$run.xfa does not exist, continuing"
    fi
done

# Aggregate all data files into one training data file
python3 aggregate_data.py ../data/training/ ../data/aggregate_data.csv

# Clean up
rm WallDistance.txt
rm GradientWallDistance.txt
rm Q1_refined.gri
rm ../data/training/training_data*

# JOB_LIST=()
# XFA_LIST=("../naca/xfa/")
# GRI_LIST=("naca_quad_q3_728.gri")
# FILE_NAME="training_data"

# for run in ${!XFA_LIST[@]}; do  
#     ../bin/xf_MeshStats -in ${XFA_LIST[run]} -wallDistance True -gWallDistance True
#     python3 process_data.py ${GRI_LIST[0]} WallDistance.txt GradientWallDistance.txt "$FILE_NAME$run.csv"
#     mv "$FILE_NAME$run.csv" ./data
# done

#Steps
#    Run Case
#    Call MeshStats
#    Pass that either to matlab or straight to python
#    Run python script

#../bin/xf_Refine -in naca_tri_q13_1584.gri -interact True, turns our gri file into one we can process