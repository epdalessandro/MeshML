import os
import sys
import numpy as np
from process_data import printTrainingData
from process_data import numInputs as numIn, numOutputs as numOut

def aggregate_data(dir: str, outFileName: str) -> None:
    """
    This function reads in the data from each file in the data directory and aggregates it into one
    input (x) and one output (y) array, which is then printed out to a user-specified file
    dir: filepath to directory to aggregate the data from
    outFileName: the name of the file where the aggregated training data will be written
    """
    filenames = os.listdir(dir) # Get all of the data files
    x, y = [], []
    for name in filenames:
        with open(dir + name) as file:
            header = file.readline()    # Process the file header
            for line in file:           # Process the data on the rest of the lines
                data = [float(num) for num in line.split(",")]
                x.append(data[0:numIn])
                y.append(data[numIn:numIn+numOut])
    printTrainingData(np.array(x), np.array(y), outFileName)

# Usage: aggregate_data.py dataDirectoryPath outputFileName
def main():
    aggregate_data(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()