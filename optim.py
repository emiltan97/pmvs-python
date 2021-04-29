import numpy as np
import cv2 as cv
from math import acos, cos, inf, sin, sqrt
from nlopt.nlopt import LN_NELDERMEAD
import nlopt
from numpy.linalg.linalg import norm, inv
from classes import Patch

VpStar = None 
ref = None
norm_ray = None
ray = None
grid = None
val = None

def run(patch, i_ref, i_VpStar) : 
    global ray 
    global norm_ray 
    global VpStar
    global ref
    global grid
    global val

    ref = i_ref 
    VpStar = i_VpStar
    ray = patch.center - ref.center
    norm_ray = norm(ray)
    ray /= norm_ray
    grid = projectGrid(patch, ref)
    val = computeGrid(ref, grid)

    lower_bounds = np.array([-inf, 0, -360])
    upper_bounds = np.array([inf, 180, 360])

    params = encode(patch)
    center, normal = decode2(params[0], params[1], params[2])
    opt = nlopt.opt(LN_NELDERMEAD, 3)
    opt.set_min_objective(myFunc)
    opt.set_maxeval(1000)
    opt.set_lower_bounds(lower_bounds)
    opt.set_upper_bounds(upper_bounds)
    opt.set_xtol_rel(1e-6)

    res = opt.optimize(params)

    center, normal = decode2(res[0], res[1], res[2])
    patch = Patch(center, normal, ref)

    # for image in VpStar : 

    #     print(projectGrid(patch,image))
    #     input("")
    # exit()

    return patch 

def computeDiscrepancy(ref, image, patch) : 
    grid1 = projectGrid(patch, ref)
    grid2 = projectGrid(patch, image)
    val1 = computeGrid(ref, grid1)
    val2 = computeGrid(image, grid2)

    return ncc(val1, val2)

def computeDiscrepancy2(image, grid_tmp) : 
    global val
    grid2 = projectGrid2(grid_tmp, image)
    val2 = computeGrid(image, grid2)

    return ncc(val, val2)

def projectGrid(patch, image) : 
    gridCoordinate = np.empty((7, 7, 3))
    margin = 7 / 2
    pmat = image.pmat
    center = pmat @ patch.center
    center /= center[2]
    dx = pmat @ (patch.center + patch.px) 
    dy = pmat @ (patch.center + patch.py) 
    dx /= dx[2]
    dy /= dy[2]
    dx -= center
    dy -= center
    scale = 1
    center /= scale;  dx /= scale;  dy /= scale
    
    left = center - dx*margin - dy*margin
    for i in range(7) : 
        temp = left
        left = left + dy
        for j in range(7) : 
            gridCoordinate[i][j] = temp
            temp = temp + dx

    return gridCoordinate

def projectGrid2(grid_tmp, image) :
    gridCoordinate = np.empty((7, 7, 3))
    for i in range(7) : 
        for j in range(7) : 
            pt = image.pmat @ grid_tmp[i][j]
            pt /= pt[2]
            gridCoordinate[i][j] = pt
    
    return gridCoordinate

def computeGrid(image, grid) : 
    val = np.empty((7, 7, 3))
    img = cv.imread(image.name)
    width = img.shape[1]
    height = img.shape[0]
    for i in range(grid.shape[0]) : 
        for j in range(grid.shape[1]) : 
            x = grid[i][j][0]
            y = grid[i][j][1]
            if (x < 0 or y < 0 or x > width - 1 or y > height - 1) : 
                val[i][j] = np.array([0, 0, 0])
            else : 
                x1  = int(x)
                x2  = int(x) + 1
                y1  = int(y)
                y2  = int(y) + 1
                q11 = img[y1][x1] 
                q12 = img[y1][x2] 
                q21 = img[y2][x1] 
                q22 = img[y2][x2] 
                val[i][j] = computeBilinearInterpolation(x, y, x1, x2, y1, y2, q11, q12, q21, q22)

    return val

def computeBilinearInterpolation(x, y, x1, x2, y1, y2, q11, q12, q21, q22) : 
    t = (x-x1) / (x2-x1)    
    u = (y-y1) / (y2-y1)

    a = q11*(1-t)*(1-u)
    b = q21*(t)*(1-u)
    c = q12*(u)*(1-t)
    d = q22*(t)*(u)

    f = a + b + c + d

    f = np.array([f[2], f[1], f[0]])

    return f 

def ncc(val1, val2) : 
    length = val1.size
    m1 = 0 
    m2 = 0 
    for i in range(val1.shape[0]) : 
        for j in range(val1.shape[1]) :
            for k in range(val1.shape[2]) : 
                m1 += val1[i][j][k]
                m2 += val2[i][j][k]
    m1 /= length
    m2 /= length
    a = 0 
    b = 0 
    c = 0
    for i in range(val1.shape[0]) : 
        for j in range(val1.shape[1]) :
            for k in range(val1.shape[2]) : 
                d1 = val1[i][j][k] - m1
                d2 = val2[i][j][k] - m2
                a += d1 * d2 
                b += d1**2 
                c += d2**2
    if b * c == 0 : 
        return 0
    res = a / sqrt(b * c)

    return res

def myFunc(opt_param, grad) :
    grid = decode(opt_param[0], opt_param[1], opt_param[2]) 

    return computeGStar(grid)

def computeGStar(grid) : 
    gStar = 0 
    for image in VpStar : 
        if image.id == ref.id : 
            continue 
        else : 
            ncc = 1 - computeDiscrepancy2(image, grid) 
            gStar += ncc
    gStar /= len(VpStar) - 1 

    return gStar

def encode(patch) : 
    # Alpha 
    ray = patch.center - ref.center
    global norm_ray 
    norm_ray = norm(ray)
    alpha = norm_ray
    # Theta
    theta = acos(patch.normal[2])
    # Phi 
    com = complex(patch.normal[0] / sin(theta), patch.normal[1] / sin(theta))
    phi = np.angle(com)

    return alpha, theta, phi

def decode(alpha, theta, phi) : 
    ret_grid = np.empty((7, 7, 4))
    for i in range(grid.shape[0]) : 
        for j in range(grid.shape[1]) : 
            pmat = ref.pmat
            W = np.array([(sin(theta) * cos(phi)), (sin(theta) * sin(phi)), (cos(theta)), 0])

            mat1 = np.array([
                [pmat[0][0], pmat[0][1], pmat[0][2]], 
                [pmat[1][0], pmat[1][1], pmat[1][2]], 
                [W[0], W[1], W[2]]
            ])
            mat2 = np.array([
                (alpha/norm_ray) * grid[i][j][0] - pmat[0][3],
                (alpha/norm_ray) * grid[i][j][1] - pmat[1][3],
                W @ (ref.center + ray*alpha)
            ])

            mat3 = inv(mat1) @ mat2
            pt = np.array([mat3[0], mat3[1], mat3[2], 1])

            ret_grid[i][j] = pt
    
    return ret_grid

def decode2(alpha, theta, phi) : 
    center = ref.center + ray * alpha
    normal = np.array([
        sin(theta) * cos(phi), 
        sin(theta) * sin(phi),
        cos(theta),
        0
    ])
    
    return center, normal