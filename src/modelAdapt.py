from subprocess import run
import tensorflow as tf
from training import normal #Import the best model, TODO: change
import process_data
from preprocess_inputs import read_data, clip_dataframe, process_dataframe, metric_exp
import sys
import numpy as np


def createBamgMetricFile(y, outFileName: str) -> None:
    with open(outFileName, "w") as fileID:
        # Print metric file header
        print(y.shape[0], y.shape[1], sep=" ", file=fileID)
        for A, B, C in y:
            print(A, B, C, sep=" ", file=fileID)

def matrixExponent(matArray: np.ndarray):
    matExponent = np.array(matArray.shape)
    for predIdx in range(matArray.shape[0]):
        A, B, C = metric_exp(matArray[predIdx,0], matArray[predIdx,1], matArray[predIdx,2])
        matExponent[predIdx] = np.array([A, B, C])
    return matExponent

def modelAdapt(checkpointDir: str, inputFile: str, metricFile: str, model: tf.keras.Model):
    latestCheckpoint = tf.train.latest_checkpoint(checkpointDir)
    model.load_weights(latestCheckpoint)

    # Get raw inputs and process them
    input_df = read_data(inputFile)
    processed_clipped = process_dataframe(clip_dataframe(input_df,1))

    # Predict metrics and save them
    predictions = model.predict(processed_clipped.to_numpy())
    metrics = matrixExponent(predictions)
    print(f"shape = {predictions.shape}\npredictions={predictions}")
    createBamgMetricFile(metrics, metricFile)

def runJob(executionDir: str, meshFilePrefix: str, jobFile: str, xfaFile: str, iteration: int) -> None:
    # Create job file from the current mesh, and flow parameters
    run(["module load matlab"])
    run(["matlab -r ../adapt/setup_runs.m"])

    # Run job from job file, then convert to gri file
    run([executionDir + "xflow", jobFile, ">", "out.txt", "&"])
    run([executionDir + "xf_Convert", "-in", xfaFile, "-out", meshFilePrefix + "-" + str(iteration) + ".gri"])
    run(["rm", "jobFile"]) # Clean up old job file

def getIterationMetrics(filename: str) -> list:
    iterationStats = [] # Should be in the format [AdaptiveIter, Iter, CFL, UFrac, Rnorm, Drag, Lift]
    with open(filename, "r") as file:
        adaptiveIterationFlag = False # Flag to tell us when we're at the start of an iteration
        prevLine = ""
        for line in file.readlines():
            if(not adaptiveIterationFlag and "Starting adaptation iteration" not in line):
                continue
            elif(not adaptiveIterationFlag and "Starting adaptation iteration" in line):
                adaptiveIteration = int(line[line.find(".")-1]) # Read the adaptiveIter from the header "Starting adaptation iteration X."
                adaptiveIterationFlag = True; continue
            else:
                if("Nonlinear solver" not in line): # Not at bottom of the solver stats printing
                    prevLine = line; continue
                else:
                    iterationStats.append([adaptiveIteration] + prevLine.split())
                    prevLine = []; adaptiveIterationFlag = False
    
    return iterationStats

def plotMetric(iterationMetrics: np.array, columns: "list[int]", labels: "list[str]", metricName: str, identifier: str):
    iterations = iterationMetrics[:][0]
    metrics = []
    # print("Plotting " + metricName)
    # plt.figure()
    # plt.plot(iterations, metrics label=labels[row])
    # plt.legend()
    # plt.xlabel("Adaptive Iteration")
    # plt.ylabel("Force")
    # plt.savefig(metricName + identifier + ".png")
    # plt.close()

def main():
    if(len(sys.argv) != 3):
        print("Usage Error; Correct Usage - modelAdapt.py numIterations checkpointDir")
        return
    
    numIterations = int(sys.argv[1])
    meshFilePrefix = "../adapt/meshData/naca"
    checkpointDir = sys.argv[2]
    inputFile = "../adapt/networkInputs.csv"
    metricFile = "../adapt/predicted.bamg.metrics"
    bamgFile = "../adapt/bamgOut.bamg"
    jobFile = "../adapt/adapt.job"
    paramFile = "../adapt/params.txt"
    execDir = "../../bin/" # directory the executables are in

    for iter in range(1, numIterations+1):
        # Use the mesh file from the previous iteration
        meshFile = meshFilePrefix + "-" + str(iter-1) + ".gri" # Use the gri file from the previous iteration
        xfaFile = meshFilePrefix + "-" + str(iter-1) + ".xfa" # Use the xfa file from the previous iteration
        run([execDir + "xf_MeshStats", "-in", xfaFile, "-wallDistance", "True", "-gWallDistance", "True"])
        with open("../input.txt", "r") as fileID:
            run([execDir + "xf_Refine", "-in", meshFile, "-interact", "True"], stdin=fileID)

        # Get the inputs from the mesh, wall distance, and gradient wall distance files
        Mesh = process_data.readgri("Q1_refined.gri")
        wDist, gDist = process_data.processWallDistances("WallDistance.txt", "GradientWallDistance.txt")
        x, _ = process_data.processMesh(Mesh, wDist, gDist, 1, paramFile)
        process_data.printData(inputFile, x)

        print(f"Iteration {iter}: Processing Mesh Finished")

        # Predict what the metrics should be
        modelAdapt(checkpointDir, inputFile, metricFile, normal)

    #     # NOTE: convert gri file to geom file (xf_Convert)
    #     # Update the mesh based on those metrics
    #     bamgFile = meshFilePrefix + "-" + str(iter) + ".bamg"
    #     xfaFile = meshFilePrefix + "-" + str(iter) + ".xfa" # Create a new xfa file for this iteration
    #     # Create the .geom file for bamg to use
    #     geomFile = meshFilePrefix + "-" + str(iter-1) + ".geom" 
    #     run([execDir + "xf_Convert", "-in", meshFile, "-out", geomFile])

    #     run(["../adapt/bamg", "-b", geomFile, "-M", metricFile, "-o", bamgFile]) # .out is a .bamg file  
    #     run([execDir + "xf_Convert", "-in", bamgFile, "-out", xfaFile]) # xfConvert -in file.bamg -out file.gri, things
    #     runJob(execDir, meshFilePrefix, jobFile, xfaFile, iter)

    #     # Clean up files
    #     run(["rm", "WallDistance.txt", "GradientWallDistance.txt", "Q1_refined.gri", inputFile, bamgFile])

    # # Post processing
    # run([execDir + "xflow", jobFile, ">", "naca.txt"])
    # iterationMetrics = getIterationMetrics("naca.txt")

if __name__ == "__main__":
    main()