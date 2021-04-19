import logging
from math import cos, pi
import numpy as np
from numpy.linalg.linalg import norm
import utils
from classes import Patch
from numpy import dot
import optim

ref = None

def run(images) : 
    print("==========================================================", flush=True)
    print("                      INITIAL MATCHING                    ", flush=True)
    print("==========================================================", flush=True)
    # P <- empty
    patches = []
    # For each image I with optical center O(I)
    for I in images : 
        print("----------", flush=True)
        print(f'IMAGE : {I.id:02d}', flush=True)
        print("----------", flush=True)
        # For each feature f detected in I 
        f_num = 1
        for f in I.feats :
            logging.info(f'IMAGE {I.id:02d}:Processing feature {f_num}....')
            # F <- {Features satisfying the epipolar constraint}
            F = computeF(I, images, f)
            # Sort F in an increasing order of distance from O(I)
            sortF(F, f)
            # For each feature f' in F
            for fprime in F :
                # Initialize c(p), n(p) and R(p) 
                global ref 
                ref = I
                p = computePatch(f, fprime)
                # Initialize V(p) and V*(p)
                Vp = computeVp(images, p)
                VpStar = computeVpStar(Vp, p, 0.6)
                if len(VpStar) >= 3 : 
                    logging.info(f'IMAGE {ref.id:02d}:Is possible patch : True')
                    # Refine c(p) and n(p)
                    new_p = refinePatch(p, VpStar)
                    # Update V(p) and V*(p)
                    Vp = computeVp(images, new_p)
                    VpStar = computeVpStar(Vp, new_p, 0.7)
                    logging.info(f'IMAGE {I.id:02d}:V*(p) size = {len(VpStar)}')
                    # If |V*(p)| < gamma
                    if len(VpStar) < 3 : 
                        # Fail
                        continue
                    # Add p to P
                    patches.append(new_p)
                    logging.info(f'IMAGE {I.id:02d}:Patch Registered.')
                    break
                else :
                    logging.info(f'IMAGE {I.id:02d}:Is possible patch : False')
            f_num += 1
    
    return patches

def computeF(I, images, f) : 
    F = [] 
    coord = np.array([f.x, f.y, 1])
    for image in images : 
        if I.id == image.id :
            continue
        else : 
            fmat = utils.fundamentalMatrix(I, image)
            epiline = fmat @ coord 
            for feat in image.feats : 
                dist = utils.distance(feat, epiline)
                if dist <= 5 : 
                    F.append(feat)
    
    return F

def sortF(F, f) : 
    for feat in F : 
        pt = utils.triangulate(f, feat, f.image.pmat, feat.image.pmat)
        vec = pt - f.image.center
        depth = norm(vec)
        feat.depth = depth
    utils.insertionSort(F)

def computePatch(f, fprime) : 
    center = utils.triangulate(f, fprime, ref.pmat, fprime.image.pmat)
    normal = ref.center - center
    normal /= norm(normal)
    patch = Patch(center, normal, ref)

    return patch

def computeVp(images, patch) : 
    Vp = [] 
    Vp.append(ref) 
    for image in images : 
        if ref.id == image.id : 
            continue 
        else : 
            angle = (dot(patch.normal, (image.center - patch.center))) / (norm(image.center - patch.center)) 
            if angle < cos(60 * pi / 180) : 
                continue 
            else : 
                Vp.append(image)
    
    return Vp

def computeVpStar(Vp, p, alpha) : 
    VpStar = []
    for image in Vp : 
        if ref.id == image.id : 
            continue 
        else :
            h = 1 - optim.computeDiscrepancy(ref, image, p)
            if h < alpha :
                VpStar.append(image) 

    return VpStar
    
def refinePatch(patch, VpStar) : 
    logging.info(f'Refining patch...')
    refinedPatch = optim.run(patch, ref, VpStar)

    return refinedPatch