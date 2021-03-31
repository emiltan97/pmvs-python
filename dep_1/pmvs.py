import argparse
import logging
import os

from datetime import datetime
from util import SIFT, applyGrid, calibrateImages, computeNeighbourImages, computePotentialFeatures, computePatch, initImages, sortPotentialFeatures, computePotentialVisibleImages, HarrisCorner

if __name__ == "__main__" : 
    # Initializing parser
    parser = argparse.ArgumentParser() 
    parser.add_argument('-f', '--filename', type=str, default="dinoSR_par.txt", help="The name of the file containing the name of the images and their corresponding camera parameters.")
    parser.add_argument('-d', '--dirname', type=str, default="C:/Users/emilt/multiview-reconstruction/src/data/sample04/dinoSparseRing/", help="The directory contain the input file.")
    parser.add_argument('-v', '--verbose', action='store_true', help="set this flag to log a verbose file")
    parser.add_argument('--display', action='store_true', help="set this flag to display the images.")
    args = parser.parse_args()
    # Verbose setting 
    if args.verbose: 
        LOG_FILENAME = datetime.now().strftime('logs/log%H%M%S%d%m%Y.log')
        logging.basicConfig(level=logging.DEBUG, filename=LOG_FILENAME, filemode='w')
    else : 
        logging.basicConfig(level=logging.INFO)
    # Registering the input images 
    os.chdir(args.dirname) # Change current directory to the directory containing the input images
    images = initImages(args.filename)
    # Calibrating the images 
    calibrateImages(images)
    # Applying grid on each image
    # applyGrid(images, 32, args.display)
    # Performing Harris Corner feature detection on each image 
    # HarrisCorner(images, args.display)
    SIFT(images, args.display)
    # For each feature in the refernce image, compute features on other images that satisfies the epipolar constraint
    for image in images : 
        features              = image.getFeatures()
        for feature in features :
            potentialFeatures = computePotentialFeatures(image, images, feature, args.display)
            potentialFeatures = sortPotentialFeatures(feature, potentialFeatures, image)
            for potentialFeature in potentialFeatures : 
                patch = computePatch(feature, potentialFeature, image)
                potentialVisibleImages = computeNeighbourImages(image, images, args.display)
                # patch.setPotentialVisibleImages(computePotentialVisibleImages(image, potentialVisibleImages, patch, 0.4))