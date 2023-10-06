from subprocess import run
import tensorflow as tf
from training import normal #Import the best model, TODO: change
import sys

def modelAdapt(checkpointFile: str, inputFile: str, model: tf.Keras.Model):
    model = tf.keras.Model()
    model.load_weights(checkpointFile)
    with open(inputFile) as input:


def main():
    if(len(sys.argv) != 3):
        print("Usage Error; Correct Usage - modelAdapt.py meshFile numIterations")
        return
    
    numIterations = sys.argv[2]
    meshFile = sys.argv[1] #"../adapt/naca.gri"
    bamgFile = "adaptedMesh"
    inputFile = "../adapt/networkInputs.csv"
    metricFile = "../adapt/networkMetrics.csv"
    paramFile = "../adapt/params.txt"
    execDir = "../../bin/" # directory the executables are in
    for iter in range(numIterations):
        run([execDir + "xf_MeshStats", "-in", meshFile, "-wallDistance", "True", "-gWallDistance", "True"])
        run([execDir + "xf_Refine", "-in", meshFile, "-interact", "True", "<", "../input.txt"])
        run(["python3", "process_data.py", "Q1_refined.gri", "WallDistance.txt", "GradientWallDistance.txt", dataFile, 1, paramFile])
        processViaModel()
        ./bamg -b meshFile -M dataFile
        run([execDir + "xf_MeshAdaptBamg", meshFile, dataFile, "bamgFile"])
        run([execDir + "xf_Convert", "-in", bamgFile, "-out", meshFile + "-" + iter + ".gri"])

        run(["rm", "WallDistance.txt", "GradientWallDistance.txt", "Q1_refined.gri", dataFile, bamgFile])

if __name__ == "__main__":
    main()