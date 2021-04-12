import argparse
import logging
import os
import preprocess

from datetime import datetime

if __name__ == "__main__" : 

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, default="dinoSR_par.txt")
    parser.add_argument('-d', '--dirname', type=str, default="C:/Users/emilt/multiview-reconstruction/src/data/sample04/dinoSparseRing2/")
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose : 
        LOG_FILENAME = datetime.now().strftime('logs/log%H%M%S%d%m%Y.log')
        logging.basicConfig(level=logging.DEBUG, filename=LOG_FILENAME, filemode='w')
    else : 
        logging.basicConfig(level=logging.INFO)
    
    os.chdir(args.dirname)

    images = preprocess.run(args.filename)
    