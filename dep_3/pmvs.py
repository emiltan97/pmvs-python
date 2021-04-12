import argparse
import logging
import os

import preProcess
import initialMatch
import matplotlib.pyplot as plt

from datetime import datetime

if __name__ == "__main__" : 
    # Initializing parser
    parser = argparse.ArgumentParser() 
    parser.add_argument('-f', '--filename', type=str, default="dinoSR_par.txt", help="The name of the file containing the name of the images and their corresponding camera parameters.")
    parser.add_argument('-d', '--dirname', type=str, default="C:/Users/emilt/multiview-reconstruction/src/data/sample04/dinoSparseRing/", help="The directory contain the input file.")
    parser.add_argument('-v', '--verbose', action='store_true', help="set this flag to log a verbose file")
    args = parser.parse_args()
    # Verbose setting 
    if args.verbose: 
        LOG_FILENAME = datetime.now().strftime('logs/log%H%M%S%d%m%Y.log')
        logging.basicConfig(level=logging.DEBUG, filename=LOG_FILENAME, filemode='w')
    else : 
        logging.basicConfig(level=logging.INFO)
    os.chdir(args.dirname)
    # Preprocessing
    images = preProcess.run(args.filename)
    # Initial matching
    patches = initialMatch.run(images)
    # Expansion 

def dispPatches(patches) :
    ax = plt.axes(projection='3d')

    xdata = []
    ydata = []
    zdata = []
    udata = []
    vdata = []
    wdata = []

    for patch in patches : 
        x = patch.centre[0]
        y = patch.centre[1]
        z = patch.centre[2]
        u = patch.normal[0]
        v = patch.normal[1]
        w = patch.normal[2]

        xdata.append(x)
        ydata.append(y)
        zdata.append(z)
        udata.append(u)
        vdata.append(v)
        wdata.append(w)

    ax.quiver(xdata, ydata, zdata, udata, vdata, wdata, length=0.1, normalize=True)
    ax.scatter3D(xdata, ydata, zdata)
    plt.show()