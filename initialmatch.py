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
import logging

ref = None
VpStar = None
iter = 1
depthVector = None

def run(images) : 
    print("==========================================================", flush=True)
    print("                      INITIAL MATCHING                    ", flush=True)
    print("==========================================================", flush=True)
    # P <- empty 
    patches = [] 
    # For each image I with optical center O(I)
    for im1 in images : 
        print("----------", flush=True)
        print(f'IMAGE : {im1.id:02d}', flush=True)
        print("----------", flush=True)
        # For each feature f detected in I
        ft_id = 1
        for feat1 in im1.feats : 
            logging.info(f'IMAGE {im1.id:02d}:Processing feature {ft_id}....')
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
                Vp = computeVp(images, patch)
                global VpStar
                VpStar = computeVpStar(Vp, patch)
                global iter 
                iter = 1
                if len(VpStar) >= 3 : 
                    logging.info(f'IMAGE {im1.id:02d}:Is possible patch : True')
                    # Refine c(p) and n(p)
                    logging.debug("BEFORE")
                    logging.debug(f'PATCH CENTER {patch.center}')
                    logging.debug(f'PATCH NORMAL {patch.normal}')
                    logging.debug(f'Vp {len(Vp)}')
                    logging.debug(f'Vp {len(VpStar)}')
                    logging.debug(f'Ref center : {ref.center}')
                    patch = refinePatch(patch)
                    logging.debug("AFTER")
                    # Update V(p) and V*(p)
                    Vp = computeVp(images, patch)
                    VpStar = computeVpStar(Vp, patch)
                    logging.debug(f'PATCH CENTER {patch.center}')
                    logging.debug(f'PATCH NORMAL {patch.normal}')
                    logging.debug(f'Vp {len(Vp)}')
                    logging.debug(f'Vp {len(VpStar)}')
                    # If |V*(p)| < gamma 
                    logging.info(f'IMAGE {im1.id:02d}:V*(p) size = {len(VpStar)}')
                    if len(VpStar) <= 3 : 
                        # Fail
                        continue
                    # Add p to P
                    patches.append(patch)
                    logging.info(f'IMAGE {im1.id:02d}:Patch Registered.')
                    # Add p to the corresponding Qj(x, y) and Qj*(x, y)
                    # Remove features from the cells where p was stored
                    # Exit innermost for loop 
                else : 
                    logging.info(f'IMAGE {im1.id:02d}:Is possible patch : False')

            ft_id += 1

    return len(patches) 

def computeF(ref, images, feat1) : 
    F = [] 
    coord = np.array([feat1.x, feat1.y, 1])
    for image in images : 
        if ref.id == image.id : 
        # if image.id != 3 : 
            continue 
        else : 
            fmat = utils.computeFundamentalMatrix(ref, image)
            # fmat = utils.computeFundamentalMatrix2(ref, image)
            epiline = fmat @ coord 
            for feat2 in image.feats : 
                dist = utils.computeDistance(feat2, epiline)
                # dist = utils.computeDistance2(feat1, feat2, fmat)
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
    normal = ref.center - cp
    normal /= norm(normal)

    # cp = np.array([-0.0093028, 0.0375282, -0.0426621, 1])
    # normal = np.array([0.400778, 0.210337, -0.891703, 0])
    patch = Patch(cp, normal, ref)
    patch.px, patch.py = utils.getPatchAxes(ref, patch)

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

# def computeVp(images) : 
#     Vp = [] 
#     Vp.append(ref)
#     for image in images : 
#         if ref.id == image.id : 
#             continue 
#         else : 
#             opticalAxis1 = np.array([
#                 ref.pmat[2][0], 
#                 ref.pmat[2][1],
#                 ref.pmat[2][2]
#             ]) 
#             opticalAxis2 = np.array([
#                 image.pmat[2][0], 
#                 image.pmat[2][1],
#                 image.pmat[2][2]
#             ]) 
#             angle = dot(opticalAxis1, opticalAxis2)
#             if angle < cos(60 * pi/180) :
#                 continue
#             else : 
#                 Vp.append(image)

#     return Vp

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
    logging.info(f'IMAGE {ref.id:02d}:Refining patch...')
    # print(f'{"Iter"}    {"X1":8s}    {"X2":8s}    {"X3":8s}', flush=True)
    print(patch.center)
    print(patch.normal)
    DoF = encode(patch) 
    # logging.debug(f'       {"X0":8s}    {"X1":8s}    {"X2":8s}')
    # logging.debug(f'DoF:{DoF[0]:8f}    {DoF[1]:8f}    {DoF[2]:8f}')
    # res = minimize(fun=myFunc, x0=DoF, method='Nelder-Mead', options={'maxfev':1000, 'xatol':1e-7}, callback=progressCallback)
    res = optim.nelder_mead(myFunc, DoF)
    print(res)
    # logging.debug(f'DoF:{res.x[0]:8f}    {res.x[1]:8f}    {res.x[2]:8f}')
    patch = decode(res[0])
    print(patch.center)
    print(patch.normal)
    exit()

    return patch

def myFunc(DoF) :
    patch = decode(DoF) 
    
    return computeGStar(patch)

def decode(DoF) :
    depthUnit = DoF[0]
    alpha = DoF[1]
    beta = DoF[2] 
    
    x = cos(alpha) * sin(beta)
    y = sin(alpha) * sin(beta)
    z = cos(beta)

    global depthVector
    depthVector = depthVector * depthUnit
    center = ref.center - depthVector
    normal = np.array([x, y, z, 0])
    patch = Patch(center, normal, None)
    px, py = getPatchAxes(ref, patch)
    patch.px = px 
    patch.py = py

    return patch

def encode(patch) :
    global depthVector
    depthVector = ref.center - patch.center
    depthUnit = norm(depthVector)
    depthVector = depthVector / depthUnit
    x = patch.normal[0]
    y = patch.normal[1]
    z = patch.normal[2]
    alpha = acos(x / sqrt(x**2 + y**2)) # yaw
    beta = acos(z / sqrt(x**2 + y**2 + z**2)) # pitch

    DoF = np.array([depthUnit, alpha, beta])

    return DoF

def computeGStar(patch) : 
    gStar = 0 
    for image in VpStar : 
        if image.id == ref.id : 
            continue 
        else : 

            # grid1 = optim.projectGrid(patch, ref.pmat, ref)
            # grid2 = optim.projectGrid(patch, image.pmat, image)
            # val1 = optim.computeGridValues(image, grid1)
            # val2 = optim.computeGridValues(image, grid2)
            # length = 75
            # m1 = 0 
            # m2 = 0 
            # for i in range(val1.shape[0]) : 
            #     for j in range(val1.shape[1]) :
            #         for k in range(val1.shape[2]) : 
            #             m1 += val1[i][j][k]
            #             m2 += val2[i][j][k]
            # m1 /= length
            # m2 /= length
            # a = 0 
            # b = 0 
            # c = 0
            # for i in range(val1.shape[0]) : 
            #     for j in range(val1.shape[1]) :
            #         for k in range(val1.shape[2]) : 
            #             d1 = val1[i][j][k] - m1
            #             d2 = val2[i][j][k] - m2
            #             a += d1 * d2 
            #             b += d1**2 
            #             c += d2**2
            #             print(grid2[i][j][k])
            # if b * c == 0 : 
            #     return 0
            # res = a / sqrt(b * c)

            # gStar += 1 - res
            gStar += 1 - optim.computeDiscrepancy(ref, image, patch) 
            # print(gStar)
            # input("")
    gStar /= len(VpStar) - 1 

    return gStar

def progressCallback(x) : 
    global iter
    print(f'{iter:4d}    {x[0]:8f}    {x[1]:8f}    {x[2]:8f}', flush=True) 
    iter += 1