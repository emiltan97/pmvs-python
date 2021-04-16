from math import cos, pi
import numpy as np
import utils
import triangulation
from numpy import dot
from math import acos, cos, sin, sqrt
from numpy.linalg import norm
from utils import getPatchAxes
from scipy.optimize import minimize
from classes import Patch
import optim
import cv2 as cv

ref = None
VpStar = None 

def run(images) : 
    # P <- empty 
    patches = [] 
    # For each image I with optical center O(I)
    for im1 in images : 
        # For each feature f detected in I
        for feat1 in im1.feats : 
            # F <- {Features satisfying the epipolar constraint}
            F = computeF(im1, images, feat1)
            # Sort F in an increasing order of distance form O(I)
            sortF(im1, feat1, F)
            # For each feature f' in F 
            for feat2 in F : 
                # Initialize c(p) n(p) and R(p) 
                global ref 
                ref = im1 
                patch = computePatch(feat1, feat2, ref)
                # Initialize V(p) and V*(p)
                Vp = computeVp(images)
                global VpStar
                VpStar = computeVpStar(Vp, patch)
                if len(VpStar) >= 3 : 
                    # Refine c(p) and n(p)
                    patch = refinePatch(patch)
                    # Update V(p) and V*(p)
                    VpStar = computeVpStar(Vp, patch)
                    # If |V*(p)| < gamma 
                    if len(VpStar) <= 3 : 
                        # Fail
                        continue
                    # Add p to P
                    patches.append(patch)
                    # Add p to the corresponding Qj(x, y) and Qj*(x, y)
                    # Remove features from the cells where p was stored
                    # Exit innermost for loop 

    return len(patches) 

def computeF(ref, images, feat1) : 
    F = [] 
    coord = np.array([feat1.x, feat1.y, 1])
    for image in images : 
        if ref.id == image.id : 
            continue 
        else : 
            fmat = utils.computeFundamentalMatrix(ref, image)
            fmat /= fmat[2][2]
            epiline = fmat @ coord 
            for feat2 in image.feats : 
                dist = utils.computeDistance(feat2, epiline)
                if dist <= 5 : 
                    # refim = cv.imread(ref.name)
                    # img = cv.imread(image.name)
                    # epiline_x = (int(-epiline[2] / epiline[0]), 0)
                    # epiline_y = (int((-epiline[2] - (epiline[1]*480)) / epiline[0]), 480)
                    # cv.line(img, epiline_x, epiline_y, (255, 0, 0), 1)
                    # cv.circle(refim, (int(feat1.x), int(feat1.y)), 4, (0, 255, 0), -1)
                    # cv.circle(img, (int(feat2.x), int(feat2.y)), 3, (0, 255, 0), -1)
                    # cv.imshow(f'Reference Image ID : {ref.id}', refim)
                    # cv.imshow(f'Sensed Image ID : {image.id}', img)
                    # cv.waitKey(0)
                    # cv.destroyAllWindows()
                    F.append(feat2)

    return F

def sortF(ref, feat1, F) : 
    for feat2 in F : 
        pt = triangulation.myVersion(feat1, feat2, ref.pmat, feat2.image.pmat)
        vec = pt - ref.center 
        depth = norm(vec)
        feat2.depth = depth
    utils.insertionSort(F)

def computePatch(feat1, feat2, ref) : 
    cp = triangulation.myVersion(feat1, feat2, ref.pmat, feat2.image.pmat)
    np = ref.center - cp
    patch = Patch(cp, np, ref)
    patch.px, patch.py = utils.getPatchAxes(ref, patch)

    return patch

def computeVp(images) : 
    Vp = [] 
    Vp.append(ref)
    for image in images : 
        if ref.id == image.id : 
            continue 
        else : 
            opticalAxis1 = np.array([
                ref.pmat[2][0], 
                ref.pmat[2][1],
                ref.pmat[2][2]
            ]) 
            opticalAxis2 = np.array([
                image.pmat[2][0], 
                image.pmat[2][1],
                image.pmat[2][2]
            ]) 
            angle = dot(opticalAxis1, opticalAxis2)
            if angle < cos(60 * pi/180) :
                continue
            else : 
                Vp.append(image)

    return Vp

def computeVpStar(Vp, patch) : 
    VpStar = []
    for image in Vp : 
        if ref.id == image.id : 
            continue
        else : 
            h = 1 - optim.computeDiscrepancy(ref, image, patch)
            if h < 0.6 : 
                VpStar.append(image)

    return VpStar

def refinePatch(patch) : 
    DoF = encode(ref, patch) 
    res = minimize(fun=funcWrapper, x0=DoF, method='Nelder-Mead', options={'maxfev':1000})
    patch = decode(res.x)

    return patch

def funcWrapper(DoF) :
    patch = decode(ref, DoF) 
    
    return computeGStar(ref, VpStar, patch)

def decode(ref, DoF) :
    depthUnit = DoF[0]
    alpha = DoF[1]
    beta = DoF[2] 
    
    x = cos(alpha) * sin(beta)
    y = sin(alpha) * sin(beta)
    z = cos(beta)

    global depthVector
    depthVector = depthVector * depthUnit
    center = ref.center + depthVector
    normal = np.array([x, y, z, 0])
    patch = Patch(center, normal, None)
    px, py = getPatchAxes(ref, patch)
    patch.px = px 
    patch.py = py

    return patch

def encode(ref, patch) :
    global depthVector
    depthVector = ref.center - patch.center
    depthUnit = norm(depthVector)
    depthVector = depthVector / depthUnit
    x = patch.normal[0]
    y = patch.normal[1]
    z = patch.normal[2]
    alpha = acos(x / sqrt(x**2 + y**2)) # yaw
    beta = acos(z / sqrt(x**2 + y**2 + z**2)) # pitch

    return depthUnit, alpha, beta

def computeGStar(ref, VpStar, patch) : 
    gStar = 0 
    for image in VpStar : 
        if image.id == ref.id : 
            continue 
        else : 
            gStar += 1 - optim.computeDiscrepancy(ref, image, patch) 
    gStar /= len(VpStar) - 1 

    return gStar
