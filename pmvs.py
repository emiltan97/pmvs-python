import argparse
import logging
import utils
import preprocess
import initialmatch
from datetime import datetime
import expansion
import os
import filter

if __name__ == "__main__" :
    # Initializing parser 
    parser =argparse.ArgumentParser()
    parser.add_argument('--filename1', type=str, default="txt/dinoSR_par.txt")
    parser.add_argument('--filename2', type=str, default="txt/features.txt")
    parser.add_argument('--filename3', type=str, default="txt/patches.txt")
    parser.add_argument('--config', type=str, default="txt/config.txt")
    parser.add_argument('--dirname', type=str, default="data/dinoSR/")
    parser.add_argument('--outname', type=str, default="out.ply")
    parser.add_argument('-n', '--numpatch', type=int, default="-1")
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-l', '--load', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')
    parser.add_argument('-a', '--auto', action='store_true')
    args = parser.parse_args()
    # Verbose settings
    if args.verbose : 
        LOG_FILENAME = datetime.now().strftime('logs/log%H%M%S%d%m%Y.log')
        logging.basicConfig(level=logging.DEBUG, filename=LOG_FILENAME, filemode='w')
    else : 
        logging.basicConfig(level=logging.INFO)
    os.chdir(args.dirname)
    # Config settings
    file = open(args.config, 'r')
    line = file.readline()
    words = line.split()
    alpha1 = float(words[0])
    alpha2 = float(words[1])
    beta = int(words[2])
    gamma = int(words[3])
    n = int(words[4])
    rho = float(words[5])
    sigma = int(words[6]) 
    omega = int(words[7])
    file.close()
    # Preprocessing 
    images = preprocess.run(args.filename1, args.filename2, beta, args.display)
    # Initial Matching
    if not args.load : 
        patches = initialmatch.run(images, alpha1, alpha2, omega, sigma, gamma, beta, args.filename3, args.display)
    else :
        patches = utils.loadPatches(images, args.filename3)
    logging.info("------------------------------------------------")
    logging.info("Writing PLY...")
    utils.writePly(patches, args.outname)
    # Iteration n=3 of expansion and filtering
    iter = 1 
    for i in range(n) : 
        print( "==========================================================")
        print(f"                        EXPANSION {iter}                        ")
        print( "==========================================================")
        expansion.run(patches, images, alpha1, alpha2, gamma, sigma, rho, beta, args.numpatch, args.filename3)
        filter.run(patches, images, sigma, gamma)
        iter += 1
        if not args.auto : 
            str = input("\nContinue? [y/n]\n")
            if str.lower() == 'n' :
                break
        else : 
            logging.info("------------------------------------------------")
            logging.info("Writing PLY...")
            utils.writePly(patches, args.outname)
    logging.info("------------------------------------------------")
    logging.info("Writing PLY...")
    utils.writePly(patches, args.outname)