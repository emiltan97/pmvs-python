from classes import Patch
from nlopt.nlopt import LN_BOBYQA, LN_NELDERMEAD
import numpy as np
import cv2 as cv
from math import acos, asin, atan, atan2, cos, inf, pi, sin, sqrt
import nlopt
from numpy.linalg.linalg import norm
import utils

g_ray = None
g_ref = None
g_VpStar = None
g_dscale = 0 
g_ascale = 0
g_center = None

def run(patch, ref, VpStar) :
    global g_VpStar
    global g_ref
    global g_ray 
    global g_center
    global g_ascale


    # center = np.array([
    #     0.00994551,
    #     0.0377007,
    #     -0.00483502,
    #     1
    # ])
    # normal = np.array([
    #     0.23121,
    #     0.194608,
    #     -0.953242, 
    #     0
    # ])
    # ref = utils.getImage(0, VpStar)
    # patch = Patch(center, normal, ref)

    # print(patch.center)
    # print(patch.normal)

    g_ref = ref
    g_VpStar = VpStar
    g_ray = patch.center - ref.center
    g_ray /= norm(g_ray)
    g_center = patch.center
    g_ascale = pi / 48

    p = encode(patch)

    # bestT = p[0]
    # bestA = p[1]
    # bestB = p[2]


    # center = np.array([
    #     0.01007091,
    #     0.03777185,
    #     -0.00515735,
    #     1
    # ])
    # normal = np.array([
    #     0.0932068,
    #     0.27464589,
    #     -0.95701731, 
    #     0
    # ])
    # ref = utils.getImage(0, VpStar)
    # patch = Patch(center, normal, ref)
    # bestG = computeGStar(patch)
    # print(bestG)


    # center = np.array([
    #     0.00999788,
    #     0.0377447,
    #     -0.00505095,
    #     1
    # ])
    # normal = np.array([
    #     0.38387,
    #     -0.207272,
    #     -0.899824, 
    #     0
    # ])
    # ref = utils.getImage(0, VpStar)
    # patch = Patch(center, normal, ref)
    # bestG = computeGStar(patch)
    # print(bestG) 
    # exit()

    # for i in range(-1, 1, 1) : 
    #     for j in range(-1, 1, 1) :
    #         for k in range(-1, 1, 1) : 
    #             if i == 0 and j == 0 and k == 0 : 
    #                 continue 
    #             tempT = p[0] + i * 2
    #             tempA = p[1] + j * pi/60
    #             tempB = p[2] + k * pi/60
    #             tempP = np.array([tempT, tempA, tempB])

    #             center, normal = decode(tempP)
    #             patch = Patch(center, normal, g_ref)
    #             gStar = computeGStar(patch)
    #             if gStar < bestG : 
    #                 bestG = gStar
    #                 bestA = tempA
    #                 bestB = tempB
    #                 bestT = tempT

    # bestP = np.array([bestT, bestA, bestB])
    # center, normal = decode(bestP)
    # refinedPatch = Patch(center, normal, g_ref)

    lower_bounds = np.array([1e-12, -inf, -inf])
    upper_bounds = np.array([inf, inf, inf])

    opt = nlopt.opt(LN_BOBYQA, 3)
    opt.set_lower_bounds(lower_bounds)
    opt.set_upper_bounds(upper_bounds)
    opt.set_maxeval(100)
    # opt.set_xtol_rel(1e-7)
    opt.set_min_objective(myFunc)

    x = []
    for i in range(3) : 
        x.append(max(min(p[i], upper_bounds[i]), lower_bounds[i]))
    
    res = opt.optimize(x)
    # res = np.array([-0.38853, -6.88612, -1.35397])
    center, normal = decode(res)    
    # print(res)
    # print(center)
    # print(normal)
    # exit()
    refinedPatch = Patch(center, normal, ref)

    return refinedPatch

def computeDiscrepancy(ref, image, patch, images) : 
    center = np.array([ 0.032469, -0.00670748, -0.0790764, 1])
    normal = np.array([0.218467, 0.28979, -0.931823, 0])
    ref = utils.getImage(3, images)
    image = utils.getImage(1, images)
    patch = Patch(center, normal, ref)
    grid1 = projectGrid(patch, ref)
    grid2 = projectGrid(patch, image)
    val1 = computeGrid(ref, grid1)
    val2 = computeGrid(image, grid2)

    print(1-ncc(val1, val2))
    exit()
    return ncc(val1, val2)

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
                # val[i][j] = img[int(y)][int(x)]
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

# def ncc(val1, val2) :
#     normalize(val1)
#     normalize(val2)

#     return dot(val1, val2)

def dot(val1, val2) : 
    size = val1.size
    res = 0 
    for i in range(val1.shape[0]) : 
        for j in range(val1.shape[1]) : 
            for k in range(val1.shape[2]) : 
                res += val1[i][j][k] * val2[i][j][k]
    
    return res / size

def normalize(val) : 
    size = val.size 
    size3 = size / 3
    ave = np.empty(3)
    for i in range(val.shape[0]) : 
        for j in range(val.shape[1]) : 
            for k in range(val.shape[2]) : 
                ave[k] += val[i][j][k]
    ave /= size3 

    ave2 = 0
    for i in range(val.shape[0]) : 
        for j in range(val.shape[1]) : 
            f0 = ave[0] - val[i][j][0]
            f1 = ave[1] - val[i][j][1]
            f2 = ave[2] - val[i][j][2]

            ave2 += f0**2 + f1**2 + f2**2

    ave2 = sqrt(ave2 / size)

    if ave2 == 0 :
        ave2 = 1

    for i in range(val.shape[0]) : 
        for j in range(val.shape[1]) : 
            for k in range(val.shape[2]) : 
                val[i][j][k] -= ave[k] 
                val[i][j][k] /= ave2

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

def decode(DoF) :
    tau = DoF[0]
    theta = DoF[1] 
    phi = DoF[2] 
    # x = cos(phi) * sin(theta)
    # y = sin(phi) * sin(theta)
    # z = cos(theta)
    z = sin(phi)
    x = abs(cos(phi)) * cos(theta)
    y = abs(cos(phi)) * sin(theta)
    center =  g_ref.center + tau * g_ray
    normal = np.array([x, y, z, 0])

    return center, normal

def encode(patch) :
    ray = patch.center - g_ref.center 
    x = patch.normal[0]
    y = patch.normal[1]
    z = patch.normal[2]

    # theta = acos(z) 
    # phi = atan2(y, x)

    theta = atan2(y, x)
    phi = atan(z / sqrt(x**2 + y**2))

    return norm(ray), theta, phi

# def encode(patch) :
#     # Encoding the patch center 
#     DoF = []
#     pmat = g_ref.pmat
#     xaxis = g_ref.xaxis
#     yaxis = g_ref.yaxis
#     zaxis = g_ref.zaxis
#     xaxis0 = np.array([xaxis[0], xaxis[1], xaxis[2], 0])
#     yaxis0 = np.array([yaxis[0], yaxis[1], yaxis[2], 0])
#     fx = xaxis0 @ pmat[0]
#     fy = yaxis0 @ pmat[1]
#     ftmp = fx + fy
#     if ftmp == 0 :
#         unit = 1
#     fz = norm(patch.center - g_ref.center)
#     unit = 2 * fz / ftmp
#     unit2 = 2 * unit
#     ray = patch.center - g_ref.center 
#     ray /= norm(ray)
#     global g_dscale
#     for image in g_VpStar : 
#         diff = image.pmat @ patch.center - image.pmat @ (patch.center - ray*unit2)
#         g_dscale += norm(diff)
#     g_dscale /= len(g_VpStar) - 1 
#     g_dscale = unit2 / g_dscale

#     DoF.append((patch.center - g_center) @ g_ray / g_dscale)
#     # Encoding the patch normal
#     normal = np.array([patch.normal[0], patch.normal[1], patch.normal[2]])
#     if patch.normal[3] != 1 and patch.normal[3]!= 0 : 
#         normal /= patch.normal[3]
    
#     fx = xaxis @ normal
#     fy = yaxis @ normal
#     fz = zaxis @ normal
#     temp2 = asin(max(-1, min(1, fy)))
#     cosb = cos(temp2)
#     if cosb == 0 : 
#         temp1 = 0 
#     else : 
#         sina = fx / cosb 
#         cosa = -fz / cosb
#         temp1 = acos(max(-1, min(1, cosa)))
#         if sina < 0 : 
#             temp1 = - temp1 
#     DoF.append(temp1 / g_ascale)
#     DoF.append(temp2 / g_ascale)

#     return DoF

# def decode(DoF) :
#     center = g_center + g_dscale * DoF[0] * g_ray
#     angle1 = DoF[1] * g_ascale
#     angle2 = DoF[2] * g_ascale

#     fx = sin(angle1) * cos(angle2)
#     fy = sin(angle2)
#     fz = -cos(angle1) * cos(angle2)

#     ftmp = g_ref.xaxis * fx + g_ref.yaxis * fy + g_ref.zaxis * fz 
#     normal = np.array([ftmp[0], ftmp[1], ftmp[2], 0])

#     return center, normal