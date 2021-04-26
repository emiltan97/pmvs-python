import logging
from math import cos, pi
import numpy as np
from numpy.linalg.linalg import norm
import utils
from classes import Patch
from numpy import dot
import optim
import cv2 as cv

def run(images, alpha1, alpha2, omega, sigma, gamma, beta, filename, isDisplay) : 
    print("==========================================================", flush=True)
    print("                     INITIAL MATCHING                     ", flush=True)
    print("==========================================================", flush=True)
    # P <- empty
    patches = []
    # For each image I with optical center O(I)
    for I in images :
        # For each feature f detected in I 
        f_num = 1
        for f in I.feats :
            # F <- {Features satisfying the epipolar constraint}
            F = computeF(I, images, f, omega, isDisplay)
            # Sort F in an increasing order of distance from O(I)
            sortF(F, f)
            # For each feature f' in F
            fp_num = 1
            for fprime in F :
                logging.info(f'IMAGE  : {I.id+1:02d}/{len(images):02d}')
                logging.info(f'FEAT   : {f_num:02d}/{len(I.feats):02d}')
                logging.info(f'FEAT\'  : {fp_num:02d}/{len(F):02d}')
                fp_num += 1
                # Initialize c(p), n(p) and R(p) 
                p = computePatch(f, fprime, I)
                # Initialize V(p) and V*(p)
                Vp = computeVp(images, p, I, sigma)
                VpStar = computeVpStar(Vp, p, alpha1, I)
                if len(VpStar) < gamma : 
                    logging.info("STATUS : FAILED")
                    logging.info("------------------------------------------------")
                    continue
                # Refine c(p) and n(p)
                new_p = refinePatch(p, VpStar, I)
                # Update V(p) and V*(p)
                Vp = computeVp(images, new_p, I, sigma)
                VpStar = computeVpStar(Vp, new_p, alpha2, I)
                # If |V*(p)| < gamma
                if len(VpStar) < gamma : 
                    # Fail
                    logging.info("STATUS : FAILED")
                    logging.info("------------------------------------------------")
                    continue
                # Add p to P
                patches.append(new_p)
                # Add p to the corresponding Qj(x, y) and Qj*(x, y)
                # Remove features from the cells where p was stored
                registerPatch(new_p, Vp, VpStar, beta, f, fprime, False)
                logging.info("STATUS : SUCCESS")
                logging.info("------------------------------------------------")
                # Exit innermost for loop 
                break
            f_num += 1
    utils.savePatches(patches, filename)

    return patches

def computeF(I, images, f, omega, isDisplay) : 
    F = [] 
    ref = cv.imread(I.name)
    coord = np.array([f.x, f.y, 1])
    if isDisplay :
        cv.circle(ref, (coord[0], coord[1]), 4, (0, 255, 0), -1)
    for image in images : 
        if I.id == image.id :
            continue
        else : 
            fmat = utils.fundamentalMatrix(I, image)
            epiline = fmat @ coord 
            for feat in image.feats :
                dist = utils.distance(feat, epiline)
                if isDisplay :
                    img = image.displayFeatureMap() 
                    epiline_x = (int(-epiline[2] / epiline[0]), 0)
                    epiline_y = (int((-epiline[2] - (epiline[1]*ref.shape[0])) / epiline[0]), ref.shape[0]) 
                    cv.line(img, epiline_x, epiline_y, (255, 0, 0), 1)
                    cv.circle(img, (feat.x, feat.y), 3, (0, 255, 0), -1)
                    cv.imshow(f'Ref : {I.id}', ref)
                    cv.imshow(f'Img : {image.id}, Dist : {dist}', img)
                    cv.waitKey(0)
                    cv.destroyAllWindows()
                if dist <= omega: 
                    F.append(feat)
    
    return F

def sortF(F, f) : 
    for feat in F : 
        pt = utils.triangulate(f, feat, f.image.pmat, feat.image.pmat)
        vec = pt - f.image.center
        depth = norm(vec)
        feat.depth = depth
    utils.insertionSort(F)

def computePatch(f, fprime, ref) : 
    center = utils.triangulate(f, fprime, ref.pmat, fprime.image.pmat)
    normal = ref.center - center
    normal /= norm(normal)
    patch = Patch(center, normal, ref)

    return patch

def computeVp(images, patch, ref, sigma) : 
    Vp = [] 
    Vp.append(ref) 
    for image in images : 
        if ref.id == image.id : 
            continue 
        else : 
            angle = (dot(patch.normal, (image.center - patch.center))) / (norm(image.center - patch.center)) 
            if angle < cos(sigma * pi / 180) : 
                continue 
            else : 
                Vp.append(image)
    
    return Vp

def computeVpStar(Vp, p, alpha, ref) : 
    VpStar = []
    VpStar.append(ref)
    for image in Vp : 
        if ref.id == image.id : 
            continue 
        else :
            h = 1 - optim.computeDiscrepancy(ref, image, p, Vp)
            if h < alpha :
                VpStar.append(image) 

    return VpStar
    
def refinePatch(patch, VpStar, ref) : 
    refinedPatch = optim.run(patch, ref, VpStar)

    return refinedPatch

def registerPatch(patch, Vp, VpStar, beta, f, fprime, isDisplay) : 
    for image in Vp : 
        pmat = image.pmat
        pt = pmat @ patch.center
        pt /= pt[2]
        x = int(pt[0]/beta) 
        y = int(pt[1]/beta)
        image.cells[y][x].q.append(patch)
        isQStar = 0
        if utils.getImage(image.id, VpStar) :
            isQStar = 1
            image.cells[y][x].qStar.append(patch)
            patch.VpStar.append(image)
        utils.removeFeatures(image, image.cells[y][x])
        cell = np.array([image.id, [x, y], isQStar, 0])
        patch.cells.append(cell)
        patch.Vp.append(image)
        if isDisplay : 
            ref = cv.imread(patch.ref.name)
            cv.circle(ref, (int(f.x), int(f.y)), 3, (0, 255, 0), -1)
            img = image.displayFeatureMap() 
            cv.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
            cv.imshow(f'Ref : {patch.ref.id}', ref)
            cv.imshow(f'Img : {image.id}', img)
            cv.waitKey(0)
            cv.destroyAllWindows()