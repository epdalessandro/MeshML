import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import math
import statistics
import sys

numInputs = 8 # Global variable for the number of inputs
numOutputs = 3 # Global variable for the number of outputs

#-----------------------------------------------------------
def readgri(fname):
    f = open(fname, "r")
    Nn, Ne, dim = [int(s) for s in f.readline().split()]
    # read vertices
    V = np.array([[float(s) for s in f.readline().split()] for n in range(Nn)])
    # read boundaries
    NB = int(f.readline())
    B = []; Bname = []
    for i in range(NB):
        s = f.readline().split(); Nb = int(s[0]); Bname.append(s[2])
        Bi = np.array([[int(s)-1 for s in f.readline().split()] for n in range(Nb)])
        B.append(Bi)

    # read elements
    Ne0 = 0; E = []
    while (Ne0 < Ne):
        s = f.readline().split(); ne = int(s[0])
        Ei = np.array([[int(s)-1 for s in f.readline().split()] for n in range(ne)])
        E = Ei if (Ne0==0) else np.concatenate((E,Ei), axis=0)
        Ne0 += ne
    f.close()
    Mesh = {'V':V, 'E':E, 'B':B, 'Bname':Bname }
    return Mesh

#-----------------------------------------------------------
def plotmesh(Mesh, fname):
    V = Mesh['V']; E = Mesh['E']; 
    f = plt.figure(figsize=(12,12))
    #plt.tripcolor(V[:,0], V[:,1], triangles=E)
    plt.triplot(V[:,0], V[:,1], E, 'k-')
    dosave = not not fname
    plt.axis('equal')
    plt.tick_params(axis='both', labelsize=12)
    f.tight_layout(); plt.show(block=(not dosave))
    if (dosave): plt.savefig(fname)
    plt.close(f)

#-----------------------------------------------------------
# Reads in the wall/gradient distances from the points on each elements and returns a per-element average distance
def processWallDistances(wName: str, gName: str):
    wallDistances, gradDistances = [], []
    with open(wName, "r") as wFile:
        for line in wFile:
            pointDistances = [float(num) for num in line.split()]
            wallDistances.append(statistics.mean(pointDistances)) # Add wallDistance average to array

    with open(gName, "r") as gFile:
        # Note: gradient is a vector so we average them separately
        for line in gFile:
            pointDistances = [float(num) for num in line.split()]
            gradX, gradY = [], []
            for idx, item in enumerate(pointDistances):
                if idx % 2 == 0:
                    gradX.append(item)
                else:
                    gradY.append(item)
            gradDistances.append([statistics.mean(gradX),statistics.mean(gradY)]) # Add gradWallDistance average to array

    return wallDistances, gradDistances

# Obtains the angle of attack and mach number from the parameters file
def getFlowParameters(num_run: int, filename: str) -> float:
    with open(filename) as fileID:
        for i in range(num_run):
            line = fileID.readline()
        mu, cosA, sinA, machNumber = line.split()
        alpha = (np.arccos(float(cosA)) + np.arcsin(float(sinA))) / 2
    return float(mu), alpha, float(machNumber)

def getReynoldsNumber(mu: float, u: float, L: float, rho: float) -> float:
    return rho*u*L/mu

# Processes the mesh to obtain the input data and the labels for training
def processMesh(Mesh: "dict[str, float]", wDist: "list[float]", gDist: "list[float]", num_run: int, paramFile: str) -> None:
    numSides = 3 # Number of sides of a triangle, no magic numbers today!
    x = np.empty([len(Mesh['E']), numInputs]) # preallocate input array
    y = np.empty([len(Mesh['E']), numOutputs]) # preallocate label array

    mu, alpha, machNumber = getFlowParameters(num_run, paramFile)
    Re = getReynoldsNumber(mu, 1, 1, 1) # u, rho = 1 as per the values used in setup_runs.m, and L = 1 due to the meshes we're using

    idx = 0
    for elem in Mesh['E']:
        centroid = [0,0]
        sides = []
        metric_system = []

        for vertex in range(numSides):
            centroid += Mesh['V'][elem[vertex]]
            x_diff, y_diff = Mesh['V'][elem[vertex]] - Mesh['V'][elem[(vertex + 1) % numSides]] # get the [x,y] difference between the vertices
            sides.append([x_diff,y_diff])
            metric_system.append([x_diff ** 2, 2 * x_diff * y_diff, y_diff ** 2])
        centroid /= numSides

        try: # If the matrix is non-singular we can use np.linalg.solve
            solution = np.linalg.solve(np.array(metric_system), np.array([1,1,1]))
        except np.linalg.LinAlgError: # Otherwise use linalg.lstsq for the least squares solution
            solution, resid, rank, s = np.linalg.lstsq(metric_system, [1,1,1], rcond=None)

        x[idx,:] = [centroid[0], centroid[1], wDist[idx], gDist[idx][0], gDist[idx][1], alpha, machNumber, math.log10(Re)]
        y[idx,:] = solution
        idx += 1
    
    return x, y

#-----------------------------------------------------------
# Print out the data in CSV format
def printData(filename: str, x: np.ndarray=np.empty(0), y: np.ndarray=np.empty(0)) -> None:
    xHeader = ["X_Pos", "Y_Pos", "Wall_Dist", "G_Wall_Dist_X", "G_Wall_Dist_Y", "AoA", "M", "Re"]
    yHeader = ["A", "B", "C"]
    header = []

    if(x.size != 0): header += xHeader
    if(y.size != 0): header += yHeader
    
    with open(filename, "w") as fileID:
        print(*header, sep=",", file=fileID) # Print the header

        if(len(header) == numInputs + numOutputs): # Both inputs
            for idx in range(x.shape[0]): print(*x[idx], *y[idx], file=fileID, sep=",")
        elif(len(header) == numInputs): # Only X input
            for idx in range(x.shape[0]): print(*x[idx], file=fileID, sep=",")
        else: # Only Y input
            for idx in range(y.shape[0]): print(*y[idx], file=fileID, sep=",")

#-----------------------------------------------------------
def plotMeshMetric(x, y, Mesh):
    # Set up figure
    f = plt.figure(figsize=(12,12))
    axis = f.add_subplot(aspect='equal')
    C = 0.55 # Scaling coefficient for ellipse size

    # Plot the mesh
    V = Mesh['V']; E = Mesh['E']; 
    plt.triplot(V[:,0], V[:,1], E, 'k-')

    # Plot the ellipses from the mesh metric
    for row in range(len(x)):
        metric = np.array([[y[row][0], y[row][1]], [y[row][1], y[row][2]]]) # Construct metric matrix from components
        w, v = np.linalg.eig(metric)
        height, width = [1.0 / (w[0] ** 0.5), 1.0 / (w[1] ** 0.5)] # Magnitudes along each axis are the inverse square root of the eigenvalues
        alpha = np.arctan2(v[1,0], v[0,0]) # get rotation from the x-axis in radians
        patch = matplotlib.patches.Ellipse([x[row][0], x[row][1]], C * height, C * width, 
                        angle=(alpha * 180.0 / math.pi), linewidth=1, fill=False) # ellipse centered on the centroid
        axis.add_patch(patch)
    plt.savefig('ellipse_plot')
    plt.close(f)

#-----------------------------------------------------------
def main():
    if(len(sys.argv) != 7):
        print("Incorrect Usage; Correct Usage - process_data meshFile wDistFile gWallDistFile outputDataFile runNum paramFile")

    meshFile = sys.argv[1]
    wDistFile = sys.argv[2]
    gWallDistFile = sys.argv[3]
    outputDataFile = sys.argv[4]
    runNum = int(sys.argv[5])
    paramFile = sys.argv[6]

    Mesh = readgri(meshFile)
    wDist, gDist = processWallDistances(wDistFile, gWallDistFile)
    x, y = processMesh(Mesh, wDist, gDist, runNum, paramFile)
    printData(outputDataFile, x, y)

    # plotmesh(Mesh, [])
    # plotMeshMetric(x,y,Mesh)

if __name__ == "__main__":
    main()

# Notes: 
# 1) Need Reynolds number to include viscosity information -- DONE
# 2) Skip stuff where runs failed and don't produce .xfa files -- DONE
# 3) Add transsonic params, then rerun cases -- DONE
# 4) Get training to work -- DONE
# 5) Try architectures
# 6) Try with/without gWallDist
# 7) Change gradient averaging to output the two values for the gradient vector --- DONE
# 8) Read up more on regression in ML
# 9) Look into stopping conditions

# Questions
# 1) How are you getting the error in the lift coefficient per iteration graphs?
# 2) How long were you running for/stopping conditions


# Do two diff networks 1 for the local params and one for the case-constant params --- DONE
# Identify where a shock is based on the stablization viscosity (StabRegVisc) --> shock centroid w/ thresholding and leading edge on airfoil
# Train a standalone neural network to do this and input it as it own (do it for a NACA 4 series)
# In demo directory there is a transsonic example, the eqn file has extra terms to get the stabilization we need (Stabilization = ...) 
    # xflow/demo/transonic