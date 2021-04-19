import numpy as np
import cv2 as cv
from math import sqrt

depthVector = None

def computeDiscrepancy(ref, image, patch) : 
    grid1 = projectGrid(patch, ref.pmat, ref)
    grid2 = projectGrid(patch, image.pmat, image)
    val1 = computeGridValues(ref, grid1)
    val2 = computeGridValues(image, grid2)

    # cv.waitKey(0)
    # cv.destroyAllWindows()  

    discrepancy = ncc2(val1, val2)

    # print("{")
    # for i in range(val1.shape[0]) : 
    #     for j in range(val1.shape[1]) : 
    #         for k in range(val1.shape[2]) : 
    #             print(f'{val1[i][j][k]}, ', end="")
    # print("}")
    # print("{")
    # for i in range(val1.shape[0]) : 
    #     for j in range(val1.shape[1]) : 
    #         for k in range(val1.shape[2]) : 
    #             print(f'{val2[i][j][k]}, ', end="")
    # print("}")

    return discrepancy

def computeGridValues(image, grid) : 
    val = np.empty((7, 7, 3))
    img = cv.imread(image.name)
    for i in range(grid.shape[0]) : 
        for j in range(grid.shape[1]) : 
            x = grid[i][j][0]
            y = grid[i][j][1]
            if (x < 0 or y < 0 or x > 480 or y > 640) : 
                val[i][j] = np.array([0, 0, 0])
            else : 
                # val[i][j] = img[x][y]
                x1  = int(x)
                x2  = int(x) + 1
                y1  = int(y)
                y2  = int(y) + 1
                q11 = img[x1][y1]
                q12 = img[x1][y2]
                q21 = img[x2][y1]
                q22 = img[x2][y2]
                val[i][j] = computeBilinearInterpolation(x, y, x1, x2, y1, y2, q11, q12, q21, q22)
                # val[i][j] = computeBilinearInterpolation2(x, y, img)
    #         print(f"({x}, {y})")
    #         print(f"[{val[i][j][0]}, {val[i][j][1]}, {val[i][j][2]}]")
    # exit()

    return val

def projectGrid(patch, pmat, image) : 
    gridCoordinate = np.empty((7, 7, 3))
    margin = 3.5

    center = pmat @ patch.center
    center /= center[2]
    dx = pmat @ (patch.center + patch.px) 
    dy = pmat @ (patch.center + patch.py) 
    dx /= dx[2]
    dy /= dy[2]
    dx -= center
    dy -= center
    
    left = center - dx*margin - dy*margin
    for i in range(7) : 
        temp = left
        left = left + dy
        for j in range(7) : 
            gridCoordinate[i][j] = temp
            temp = temp + dx
    # img = cv.imread(image.name)
    # cv.rectangle(img, (int(gridCoordinate[0][0][0]), int(gridCoordinate[0][0][1])), (int(gridCoordinate[4][4][0]), int(gridCoordinate[4][4][1])), (0, 255, 0), -1)
    # cv.imshow(f'{image.name}', img)

    return gridCoordinate

def ncc(val1, val2) : 
    length = 75
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

def ncc2(val1, val2) :

    val1 = normalize(val1)
    val2 = normalize(val2)

    size = 147 
    ans = 0 
    for i in range (size) : 
        ans += val1[i] * val2[i]
    
    ans /= size 

    return ans

def normalize(val) : 
    size = 147
    size3 = int(size /3)

    temp = [] 
    for i in range(val.shape[0]) : 
        for j in range(val.shape[1]) :
            for k in range(val.shape[2]) : 
                temp.append(val[i][j][k])

    ave = np.array([0, 0, 0])
    num = 0
    for i in range(size3) :
        ave[0] += temp[num]
        num += 1
        ave[1] += temp[num]
        num += 1
        ave[2] += temp[num]
        num += 1
    ave = ave / size3 
    ave2 = 0
    num = 0
    for i in range(size3) : 
        f0 = ave[0] - temp[num]
        num += 1
        f1 = ave[1] - temp[num]
        num += 1
        f2 = ave[2] - temp[num]
        num += 1

        ave2 += f0*f0 + f1*f1 + f2*f2

    ave2 = sqrt(ave2 / size)
    if ave2 == 0 : 
        ave2 = 1 

    num = 0
    for i in range(size3) :
        temp[num] -= ave[0] 
        temp[num] /= ave2 
        num += 1 
        temp[num] -= ave[1] 
        temp[num] /= ave2 
        num += 1 
        temp[num] -= ave[2] 
        temp[num] /= ave2 
        num += 1 

    return temp

import copy

'''
    Pure Python/Numpy implementation of the Nelder-Mead algorithm.
    Reference: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
'''


def nelder_mead(f, x_start,
                step=0.1, no_improve_thr=10e-6,
                no_improv_break=10, max_iter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    '''
        @param f (function): function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x_start
        @param x_start (numpy array): initial position
        @param step (float): look-around radius in initial step
        @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
            an improvement lower than no_improv_thr
        @max_iter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
        @alpha, gamma, rho, sigma (floats): parameters of the algorithm
            (see Wikipedia page for reference)
        return: tuple (best parameter array, best score)
    '''

    # init
    dim = len(x_start)
    prev_best = f(x_start)
    no_improv = 0
    res = [[x_start, prev_best]]

    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step
        score = f(x)
        res.append([x, score])

    # simplex iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        # break after max_iter
        if max_iter and iters >= max_iter:
            return res[0]
        iters += 1

        # break after no_improv_break iterations with no improvement
        print(best)

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0]

        # centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = f(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            escore = f(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = f(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            score = f(redx)
            nres.append([redx, score])
        res = nres

def computeBilinearInterpolation2(x, y, img) : 
    lx = int(x) 
    ly = int(y) 
    dx1 = x - lx
    dy1 = y - ly
    dx0 = 1 - dx1 
    dy0 = 1 - dy1 

    dx0 = 0.0752869
    dx1 = 0.924713
    dy0 = 0.957916
    dy1 = 0.0420837

    f00 = dx0*dy0
    f01 = dx0*dy1
    f10 = dx1*dy0
    f11 = dx1*dy1
    r = 0
    g = 0 
    b = 0 
    val1 = img[lx][ly]
    val2 = img[lx+1][ly+1]
    val3 = img[lx][ly+1]
    val4 = img[lx+1][ly]

    r += 8.214 + 0.355223
    g += val1[1]*f00 + val3[1] * f01
    b += val1[2]*f00 + val3[2] * f01
    r += val2[0]*f10 + val4[0] * f11
    g += val2[1]*f10 + val4[1] * f11
    b += val2[2]*f10 + val4[2] * f11

    print(val1)
    print(val2)
    print(val3)
    print(val4)

    return (r, g, b)

def computeBilinearInterpolation(x, y, x1, x2, y1, y2, q11, q12, q21, q22) : 
    t = (x-x1) / (x2-x1)    
    u = (y-y1) / (y2-y1)

    a = q11*(1-t)*(1-u)
    b = q21*(t)*(1-u)
    c = q12*(u)*(1-t)
    d = q22*(t)*(u)

    f = a + b + c + d

    return f 