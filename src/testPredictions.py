from models.normal import normal
from subprocess import run
import process_data
import tensorflow as tf
import numpy as np
from preprocess_inputs import read_data, clip_dataframe, process_dataframe

meshFilePrefix = "naca"
checkpointFile = "../checkpoints/normalBestWeights.ckpt.index"
inputFile = "../adapt/networkInputs.csv"
metricFile = "../adapt/predicted.bamg.metrics"
bamgFile = "../adapt/bamgOut.bamg"
jobFile = "../adapt/adapt.job"
paramFile = "../adapt/params.txt"
execDir = "../../bin/" # directory the executables are in

meshFile = "../adapt/naca.gri"
xfaFile = "../adapt/naca_A00.xfa"
run([execDir + "xf_MeshStats", "-in", xfaFile, "-wallDistance", "True", "-gWallDistance", "True"])
with open("../input.txt", "r") as fileID:
    run([execDir + "xf_Refine", "-in", meshFile, "-interact", "True"], stdin=fileID)

# Get the inputs from the mesh, wall distance, and gradient wall distance files
Mesh = process_data.readgri("Q1_refined.gri")
wDist, gDist = process_data.processWallDistances("WallDistance.txt", "GradientWallDistance.txt")
x, y = process_data.processMesh(Mesh, wDist, gDist, 1, paramFile)
process_data.printData(inputFile, x)
print(f"xShape = {np.array(x).shape}")
print(f"yShape = {np.array(y).shape}")
input_df = read_data(inputFile)
processed_clipped = process_dataframe(clip_dataframe(input_df,1))

# Predict what the metrics should be
# modelAdapt(checkpointFile, inputFile, metricFile, normal)
predictUntrained = normal.predict(x)
print(f"predictedU = {predictUntrained}")
print(type(predictUntrained))

evalOutputUntrained = normal.evaluate(processed_clipped, y)
print(f"evalOutputUntrained = {evalOutputUntrained}")

latest = tf.train.latest_checkpoint("../checkpoints")
normal.load_weights(latest)
# predictTrained = normal.predict(x)
evalOutputTrained = normal.evaluate(processed_clipped, y)
print(f"evalOutputTrained = {evalOutputTrained}")