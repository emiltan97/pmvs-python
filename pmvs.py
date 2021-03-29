import argparse
import logging
import os

import preProcess
import initialMatch

from datetime import datetime

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
    os.chdir(args.dirname)
    # Preprocessing 
    images = preProcess.run(filename=args.filename, gridSize=2, isDisplay=args.display)
    # Initial matching
    initialMatch.run(images, args.display)
