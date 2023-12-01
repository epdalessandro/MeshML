#!/bin/bash
models=("model", "paperModel", "biggerH1H2", "biggerH3", "biggerHidden")
for ((exp = 9; exp <= 11; exp++)); do
    # for name in ${models[@]}; do
    #     touch $name$((2**exp))_data.csv
    # done
    time python3 training.py ../data/9_21_23/processed_clipped_data.csv $((2**exp))
    mv *.png ./output
    mv *.csv ./output
done