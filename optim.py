from classes import Patch
from nlopt.nlopt import LN_BOBYQA
import numpy as np
import cv2 as cv
from math import acos, asin, cos, inf, pi, sin, sqrt
import nlopt
from numpy.linalg.linalg import norm

g_dscale = 0 
g_ascale = 0
g_center = None
g_ray = None
g_ref = None
g_VpStar = None

def run(patch, ref, VpStar) :
    global g_VpStar
    global g_center
    global g_ref
    global g_ray 
    global g_ascale

    g_ref = ref
    g_VpStar = VpStar
    g_center = patch.center
    g_ray = patch.center - ref.center
    g_ray /= norm(g_ray)
    g_ascale = pi / 48

    p = encode(patch)
    min_angle = -23.99999
    max_angle =  23.99999
    lower_bounds = np.array([-inf, min_angle, min_angle])
    upper_bounds = np.array([inf, max_angle, max_angle])

    opt = nlopt.opt(LN_BOBYQA, 3)
    opt.set_lower_bounds(lower_bounds)
    opt.set_upper_bounds(upper_bounds)
    opt.set_maxeval(1000)
    opt.set_xtol_rel(1.e-7)
    opt.set_min_objective(myFunc)

    x = []
    for i in range(3) : 
        x.append(max(min(p[i], upper_bounds[i]), lower_bounds[i]))
    
    res = opt.optimize(x)
    center, normal = decode(res)
    refinedPatch = Patch(center, normal, ref)

    return refinedPatch


def computeDiscrepancy(ref, image, patch) : 
    grid1 = projectGrid(patch, ref)
    grid2 = projectGrid(patch, image)
    val1 = computeGrid(ref, grid1)
    val2 = computeGrid(image, grid2)

    return ncc(val1, val2)

def projectGrid(patch, image) : 
    gridCoordinate = np.empty((5, 5, 3))
    margin = 5 / 2
    pmat = image.pmat
    center = pmat @ patch.center
    center /= center[2]
    dx = pmat @ (patch.center + patch.px) 
    dy = pmat @ (patch.center + patch.py) 
    dx /= dx[2]
    dy /= dy[2]
    dx -= center
    dy -= center
    
    left = center - dx*margin - dy*margin
    for i in range(5) : 
        temp = left
        left = left + dy
        for j in range(5) : 
            gridCoordinate[i][j] = temp
            temp = temp + dx

    return gridCoordinate

def computeGrid(image, grid) : 
    val = np.empty((5, 5, 3))
    img = cv.imread(image.name)
    width = img.shape[0]
    height = img.shape[1]
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
                q11 = img[x1][y1]
                q12 = img[x1][y2]
                q21 = img[x2][y1]
                q22 = img[x2][y2]
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

def myFunc(DoF, grad) :
    center, normal = decode(DoF) 
    patch = Patch(center, normal, g_ref)

    return computeGStar(patch)

def computeGStar(patch) : 
    gStar = 0 
    for image in g_VpStar : 
        if image.id == g_ref.id : 
            continue 
        else : 
            ncc = 1 - computeDiscrepancy(g_ref, image, patch) 
            gStar += ncc
    gStar /= len(g_VpStar) - 1 

    return gStar

def encode(patch) :
    # Encoding the patch center 
    DoF = []
    pmat = g_ref.pmat
    xaxis = g_ref.xaxis
    yaxis = g_ref.yaxis
    zaxis = g_ref.zaxis
    xaxis0 = np.array([xaxis[0], xaxis[1], xaxis[2], 0])
    yaxis0 = np.array([yaxis[0], yaxis[1], yaxis[2], 0])
    fx = xaxis0 @ pmat[0]
    fy = yaxis0 @ pmat[1]
    ftmp = fx + fy
    if ftmp == 0 :
        unit = 1
    fz = norm(patch.center - g_ref.center)
    unit = 2 * fz / ftmp
    unit2 = 2 * unit
    ray = patch.center - g_ref.center 
    ray /= norm(ray)
    global g_dscale
    for image in g_VpStar : 
        diff = image.pmat @ patch.center - image.pmat @ (patch.center - ray*unit2)
        g_dscale += norm(diff)
    g_dscale /= len(g_VpStar) - 1 
    g_dscale = unit2 / g_dscale

    DoF.append((patch.center - g_center) @ g_ray / g_dscale)
    # Encoding the patch normal
    normal = np.array([patch.normal[0], patch.normal[1], patch.normal[2]])
    if patch.normal[3] != 1 and patch.normal[3]!= 0 : 
        normal /= patch.normal[3]
    
    fx = xaxis @ normal
    fy = yaxis @ normal
    fz = zaxis @ normal
    temp2 = asin(max(-1, min(1, fy)))
    cosb = cos(temp2)
    if cosb == 0 : 
        temp1 = 0 
    else : 
        sina = fx / cosb 
        cosa = -fz / cosb
        temp1 = acos(max(-1, min(1, cosa)))
        if sina < 0 : 
            temp1 = - temp1 
    DoF.append(temp1 / g_ascale)
    DoF.append(temp2 / g_ascale)

    return DoF

def decode(DoF) :
    center = g_center + g_dscale * DoF[0] * g_ray
    angle1 = DoF[1] * g_ascale
    angle2 = DoF[2] * g_ascale

    fx = sin(angle1) * cos(angle2)
    fy = sin(angle2)
    fz = -cos(angle1) * cos(angle2)

    ftmp = g_ref.xaxis * fx + g_ref.yaxis * fy + g_ref.zaxis * fz 
    normal = np.array([ftmp[0], ftmp[1], ftmp[2], 0])

    return center, normal