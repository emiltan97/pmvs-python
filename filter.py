from optim import computeGStar
import utils
from initialmatch import computeVp

def run(patches, images, sigma, gamma) : 
    for patch in patches : 
        if filter1(patch, images) or filter2(patch, sigma, gamma) or filter3(patch, images) : 
            patches.remove(patch)
    
    return patches

def filter1(patch, images) : 
    U = []
    for cell in patch.cells : 
        image = utils.getImage(cell[0], images)
        i_cell = image.cells[cell[1][1]][cell[1][0]]
        for p in i_cell.q : 
            if utils.isNeighbour(patch, p, 2) :
                continue
            U.append(p)
    
    a = len(patch.VpStar)
    b = 1 - utils.computeGStar(patch)
    c = 0 
    for pi in U : 
        c += 1 - computeGStar(pi)
    
    if a * b < c : 
        return True
    return False

def filter2(patch, sigma, gamma) : 
    images = computeVp(patch.VpStar, patch, patch.ref, sigma)
    if len(images) < gamma : 
        return True
    return False

def filter3(patch, images) : 
    patches = []
    for cell in patch.cells : 
        image = utils.getImage(cell[0], images)
        C = [
            image.cells[cell[1][1]][cell[1][0]], 
            image.cells[cell[1][1] + 1][cell[1][0]],
            image.cells[cell[1][1] - 1][cell[1][0]],
            image.cells[cell[1][1]][cell[1][0] + 1],
            image.cells[cell[1][1]][cell[1][0] - 1]
        ]
        for c in C : 
            for p in c.q : 
                patches.append(p)
    
    pos = 0

    for p in patches : 
        if utils.isNeighbour(patch, p, 2) : 
            pos += 1
    
    ratio = pos / len(patches) * 100

    if ratio < 0.25 : 
        return True
    return False
