import numpy as np
from numpy.core.defchararray import center

from numpy.linalg import pinv, norm
from numpy import dot, cross 

def mainVersion(ref, patch) :
    pm = ref.projectionMatrix
    pm /= pm[2][3]
    # normal = patch.normal / patch.normal[3]
    normal = patch.normal
    zaxis = np.array([
        normal[0],
        normal[1],
        normal[2]
    ])
    xaxis = np.array([
        ref.extrinsic[0][0],
        ref.extrinsic[0][1],
        ref.extrinsic[0][2]
    ])
    yaxis = cross(zaxis, xaxis)
    yaxis /= norm(yaxis)
    xaxis = cross(yaxis, zaxis)
    depth = norm(patch.centre - ref.opticalCentre)
    scale = 2 * depth / ((ref.intrinsic[0][0]) + (ref.intrinsic[1][1])) 
    px = np.array([
        xaxis[0],
        xaxis[1],
        xaxis[2],
        0
    ])
    py = np.array([
        yaxis[0],
        yaxis[1],
        yaxis[2],
        0
    ])
    px *= scale
    py *= scale
    xdis = norm((pm @ (patch.centre + px)) - (pm @ patch.centre))
    ydis = norm((pm @ (patch.centre + py)) - (pm @ patch.centre))
    px /= xdis
    py /= ydis

    print(ref.opticalCentre)
    print(patch.centre)
    print(normal)
    print(px)
    print(py)

    exit()

    return px, py

def myVersion(ref, patch) :
    projectionMatrix = ref.projectionMatrix
    projectionMatrix /= projectionMatrix[2][3]
    projectionMatrixPlus = pinv(projectionMatrix)
    capitalX = patch.centre[0]
    smallx = projectionMatrix @ patch.centre
    smallx /= smallx[2]
    lamda = (capitalX - ((projectionMatrixPlus[0]) @ (smallx))) / (ref.opticalCentre[0])
    smallx = np.array([
        smallx[0] - 2,
        smallx[1] - 2,
        smallx[2]
    ])
    capitalX = projectionMatrixPlus @ smallx + lamda * ref.opticalCentre
    pt = projectionMatrix @ capitalX 
    ct = projectionMatrix @ patch.centre


def goLangVersion(ref, patch) : 
    projectionMatrix = ref.projectionMatrix
    normal = patch.normal
    centre = patch.centre
    inv = pinv(projectionMatrix)
    scaleR = (inv[0][0] * normal[0]) + (inv[1][0] * normal[1]) + (inv[2][0] * normal[2])
    scaleU = (inv[0][1] * normal[0]) + (inv[1][1] * normal[1]) + (inv[2][1] * normal[2])
    right = np.array([inv[0][0]-scaleR*normal[0], inv[1][0]-scaleR*normal[1], inv[2][0]-scaleR*normal[2], 0])
    up = np.array([inv[0][1]-scaleU*normal[0], inv[1][1]-scaleU*normal[1], inv[2][1]-scaleU*normal[2], 0])
    scale = dot(centre, projectionMatrix[2])
    right *= scale
    up *= scale

    i = -2
    while i <= 2 :
        j = -2 
        while j <= 2 :
            gridPt = patch.centre + i*right + j*up
            projectedGridPt = projectionMatrix @ gridPt
            projectedGridPt /= projectedGridPt[2]
            j += 1
        i += 1

    print(patch.centre)
    print(normal)
    print(right)
    print(up)

    exit()

def yasuVersion(ref, patch) : 
    pmat= ref.projectionMatrix
    mzaxes = np.array([pmat[2][0], pmat[2][1], pmat[2][2]])
    mxaxes = np.array([pmat[0][0], pmat[0][1], pmat[0][2]])
    myaxes = cross(mzaxes, mxaxes)
    myaxes /= norm(myaxes)
    mxaxes = cross(myaxes, mzaxes)
    xaxe = np.array([mxaxes[0], mxaxes[1], mxaxes[2], 0])
    yaxe = np.array([myaxes[0], myaxes[1], myaxes[2], 0])
    fx = xaxe @ pmat[0]
    fy = yaxe @ pmat[1]
    fz = norm(patch.centre - ref.opticalCentre)
    ftmp = fx + fy
    pscale = fz / ftmp
    # pscale = 2 * 2 * fz / ftmp

    normal3 = np.array([patch.normal[0], patch.normal[1], patch.normal[2]])
    yaxis3 = cross(normal3, mxaxes)
    yaxis3 /= norm(yaxis3)
    xaxis3 = cross(yaxis3, normal3)
    
    pxaxis = np.array([xaxis3[0], xaxis3[1], xaxis3[2], 0])
    pyaxis = np.array([yaxis3[0], yaxis3[1], yaxis3[2], 0])

    pxaxis *= pscale
    pyaxis *= pscale 

    # pmat = np.array([
    #     [235.956, 771.439, -200.993, 43.3371],
    #     [498.067, 26.5942, 667.023, 65.8441],
    #     [0.808701, -0.279553, -0.517545, 0.667882]
    # ])
    # pmattmp = pmat / 2
    # pmat = np.array([pmattmp[0], pmattmp[1], pmat[2]])

    a = pmat @ (patch.centre + pxaxis)
    b = pmat @ (patch.centre + pyaxis)
    c = pmat @ (patch.centre)
    a /= a[2]
    b /= b[2]
    c /= c[2]

    xdis = norm(a - c)
    ydis = norm(b - c)

    pxaxis /= xdis 
    pyaxis /= ydis

    return pxaxis, pyaxis

def ogVersion(ref, patch) : 
    normal = np.array([
        patch.normal[0],
        patch.normal[1],
        patch.normal[2]
    ])
    # normal = patch.normal / patch.normal[3]
    # normal = np.array([
    #     patch.normal[0],
    #     patch.normal[1],
    #     patch.normal[2]
    # ])
    projectionMatrix = ref.projectionMatrix
    cameraXAxis = np.array([
        projectionMatrix[0][0],
        projectionMatrix[0][1],
        projectionMatrix[0][2],
    ])
    cameraYAxis = np.array([
        projectionMatrix[1][0],
        projectionMatrix[1][1],
        projectionMatrix[1][2],
    ])
    cameraXAxis /= norm(cameraXAxis)
    cameraYAxis /= norm(cameraYAxis)
    y = cross(cameraXAxis, normal) / norm(cross(cameraXAxis, normal))
    x = cross(cameraYAxis, normal) / norm(cross(cameraYAxis, normal))
    s = ((2)*(norm(patch.centre - ref.opticalCentre))) / (((np.array([cameraXAxis[0], cameraXAxis[1], cameraXAxis[2], 0])) @ (projectionMatrix[0][0], projectionMatrix[0][1], projectionMatrix[0][2], 1)) + ((np.array([cameraYAxis[0], cameraYAxis[1], cameraYAxis[2], 0])) @ (projectionMatrix[1][0], projectionMatrix[1][1], projectionMatrix[1][2], 1)))
    alpha = np.array([
        x[0],
        x[1],
        x[2],
        0
    ])
    beta = np.array([
        y[0],
        y[1],
        y[2],
        0
    ])
    px = (alpha * s) / norm(((projectionMatrix) @ (patch.centre+alpha)) - ((projectionMatrix) @ (patch.centre)))
    py = (beta * s) / norm(((projectionMatrix) @ (patch.centre+beta)) - ((projectionMatrix) @ (patch.centre)))

    print(ref.opticalCentre)
    print(patch.centre)
    print(normal)
    print(px)
    print(py)

    exit()

def jiaChenVersion(ref, patch) :
    # Compute x and y vectors lying on the patch
    projectionMatrix = ref.projectionMatrix

    # opticalAxis = projectionMatrix[2]
    # opticalAxis[3] = 0 
    # ftmp = norm(opticalAxis)
    # opticalAxis[3] = projectionMatrix[2][3]
    # opticalAxis /= ftmp
    # cameraZAxis = np.array([
    #     opticalAxis[0],
    #     opticalAxis[1],
    #     opticalAxis[2]
    # ])
    cameraXAxis = np.array([
        projectionMatrix[0][0],
        projectionMatrix[0][1],
        projectionMatrix[0][2]
    ])
    # cameraYAxis = cross(cameraZAxis, cameraXAxis)
    # cameraYAxis /= norm(cameraYAxis)
    # cameraXAxis = cross(cameraYAxis, cameraZAxis)

    normal = patch.normal / patch.normal[3]
    normal = np.array([
        patch.normal[0], 
        patch.normal[1],
        patch.normal[2]
    ])
    patchYAxis = cross(cameraXAxis, normal)
    patchXAxis = cross(patchYAxis, normal)
    patchXAxis = np.array([
        patchXAxis[0],
        patchXAxis[1],
        patchXAxis[2],
        0
    ])
    patchYAxis = np.array([
        patchYAxis[0],
        patchYAxis[1],
        patchYAxis[2],
        0,
    ])
    # Compute the bilinear interpolated values at the projection of the grid point
    i = -2
    while i <= 2 :
        j = -2 
        while j <= 2 :
            gridPt = patch.centre + i*patchXAxis + j*patchYAxis
            projectedGridPt = projectionMatrix @ gridPt
            projectedGridPt /= projectedGridPt[2]
            j += 1
            print(projectedGridPt)
        i += 1
    exit()

    return 