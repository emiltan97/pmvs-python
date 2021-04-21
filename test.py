from numpy.linalg.linalg import inv, pinv
from classes import Patch
import preprocess 
import numpy as np
import utils
import matplotlib.pyplot as plt

images = preprocess.run("dinoSR_par.txt", "C:/Users/emilt/multiview-reconstruction/src/data/sample04/dinoSparseRing2/")

def loadPatches(images) : 
    file = open("patches.txt", 'r')
    line = file.readline()
    words = line.split() 
    center = np.empty(4)
    normal = np.empty(4)

    for i in range(len(center)) :   
        center[i] = float(words.pop(0))
    
    for i in range(len(normal)) : 
        normal[i] = float(words.pop(0))
    
    ref = utils.getImage(int(words.pop(0)), images)

    patch = Patch(center, normal, ref)

    ids = []

    while words : 
        id = int(words.pop(0))
        ids.append(id)
        x = int(words.pop(0))
        y = int(words.pop(0))
        cell = np.array([id, [x, y]])
        patch.cells.append(cell)

    VpStar = []
    while ids : 
        VpStar.append(utils.getImage(ids.pop(0), images))            

    patch.VpStar = VpStar

    return patch

patch = loadPatches(images)


xdata = [] 
ydata = []
zdata = []

# xdata.append(patch.ref.center[0])
# ydata.append(patch.ref.center[1])
# zdata.append(patch.ref.center[2])
xdata.append(patch.center[0])
ydata.append(patch.center[1])
zdata.append(patch.center[2])

for feat in patch.ref.feats : 
    # x = np.array([feat.x, feat.y, 1])
    # P = patch.ref.pmat
    # C = patch.ref.center
    # X = pinv(P) @ x - 0.5 * C
    # X /= X[3]

    A = np.empty((3, 3))
    b = np.array([feat.x, feat.y, 1])
    for y in range(3) : 
        for x in range(3) :
            A[y][x] = patch.ref.pmat[y][x]
        b[y] -= patch.ref.pmat[y][3]
    IA = inv(A)
    X = IA @ b 
    X /= X[3]
    ax = plt.axes(projection='3d')
    ax.quiver(patch.ref.center[0], patch.ref.center[1], patch.ref.center[2], X[0], X[1], X[2], length=5)

    ax.scatter3D(xdata, ydata, zdata)
    plt.show()