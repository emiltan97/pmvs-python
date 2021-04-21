from math import cos, pi
from numpy.linalg.linalg import norm
from initialmatch import computeVpStar, refinePatch, registerPatch
from classes import Patch
import utils
from numpy import dot
import logging

def run(P, images, alpha1, alpha2, gamma, sigma, rho, filename) : 
    p_num = 1 
    # While P is not empty
    while len(P) > 0 : 
        # Pick and remove a patch p from P 
        p = P.pop(0)
        # For each image cell Ci(x, y) containing p
        c_num = 1
        for cell in p.cells :
            C = collectCells(cell, p, images, rho, alpha1)
            ci_num = 1
            for ci in C :
                logging.info(f'PATCH  : {p_num:02d}/{len(P):02d}')
                logging.info(f'CELL   : {c_num:02d}/{len(p.cells):02d}')
                logging.info(f'Ci     : {ci_num:02d}/{len(C):02d}')
                ci_num += 1 
                # Create a new patch candidate pprime
                pprime = reconstructPatch(p, ci)
                Vp = p.VpStar
                # Update V*(p')
                VpStar = computeVpStar(Vp, pprime, alpha1, pprime.ref)
                if len(VpStar) <= 1 :
                    logging.info("STATUS : FAILED")
                    logging.info("------------------------------------------------")
                    continue 
                # Refine c(p') and n(p')
                new_pprime = refinePatch(pprime, VpStar, pprime.ref)
                # Add visible images (a depth-map test) to V(p')
                addImages(new_pprime, images, Vp, sigma)
                # Update V*(p')
                VpStar = computeVpStar(Vp, new_pprime, alpha2, pprime.ref)
                # If |V*(p')| < gamma 
                if len(VpStar) < gamma : 
                    # Go back to For loop (failure)pprime.ref
                    logging.info("STATUS : FAILED")
                    logging.info("------------------------------------------------")
                    continue 
                # Add p' to P
                P.append(new_pprime)
                # Add p' to corresponding Qj(x, y) and Q*j(x, y)
                registerPatch(new_pprime, VpStar)
                logging.info("STATUS : SUCCESS")
                logging.info("------------------------------------------------")
                utils.savePatch(new_pprime, filename)
            c_num += 1
        p_num += 1

def collectCells(cell, patch, images, rho, alpha) :
    id = cell[0]
    x = cell[1][0]
    y = cell[1][1]
    image = utils.getImage(id, images)
    C = []

    c1 = image.cells[x-1][y]
    c2 = image.cells[x+1][y]
    c3 = image.cells[x][y-1]
    c4 = image.cells[x][y+1]

    if utils.identifyCell(c1, image, patch, rho, alpha) : 
        C .append(c1)
    if utils.identifyCell(c2, image, patch, rho, alpha) : 
        C .append(c2)
    if utils.identifyCell(c3, image, patch, rho, alpha) : 
        C .append(c3)
    if utils.identifyCell(c4, image, patch, rho, alpha) : 
        C .append(c4)
    
    return C 

def reconstructPatch(p, ci) :
    np = p.normal
    ref = p.ref
    cp = utils.computeCenter(p, ci)

    pprime = Patch(cp, np, ref)

    return pprime 

def addImages(patch, images, Vp, sigma) :
    for image in images :
        if utils.hasImage(image, Vp) :
           continue
        else :
            angle = (dot(patch.normal, (image.center - patch.center))) / (norm(image.center - patch.center)) 
            if angle < cos(sigma * pi / 180) : 
                continue 
            else : 
                Vp.append(image)