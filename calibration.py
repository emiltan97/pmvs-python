import numpy as np
import os

def estimateProjectionMatrix(x, y, P, num) : 
    Q = np.array([])

    for i in range(num) : 
        Pk      = P[:, i]
        zeroMat = np.zeros(4)
        xk      = x[i]
        yk      = y[i] 
        Qk      = np.array([
            np.hstack((Pk.T, zeroMat, (-xk) * (Pk.T))),
            np.hstack((zeroMat, Pk.T, (-yk) * (Pk.T)))
        ])
        if np.array_equal(Q, []) : 
            Q = Qk
        else : 
            Q = np.vstack((Q, Qk))

    U, E, V = np.linalg.svd(Q)

    res = np.array([
        [V[11, 0], V[11, 1], V[11, 2],  V[11, 3]],
        [V[11, 4], V[11, 5], V[11, 6],  V[11, 7]],
        [V[11, 8], V[11, 9], V[11, 10], V[11, 11]]
    ])

    return res 

if __name__ == "__main__" : 
    path = 'data/mydata/txt/datapoints/'

    i = 0
    for filename in os.listdir(path):
        file = open(path + filename, 'r')
        lines = file.readlines()
        p_num = int(lines.pop(0))
        x = []
        y = []
        p = np.array([])
        for j in range(p_num) :
            line = lines.pop(0)
            words = line.split()
            x.append(int(words[0]))
            y.append(int(words[1]))
            pt = np.array([float(words[2]) / 1000, float(words[3]) / 1000, float(words[4]) / 1000, float(words[5])])
            if np.array_equal(p, []) :
                p = pt 
            else : 
                p = np.vstack((p, pt))
        p = p.T
        file.close()
        pmat = estimateProjectionMatrix(x, y, p, p_num)

        pt = np.array([0, 0, 0, 1])
        res = pmat @ pt 
        print(res)
        res /= res[2] 
        print(res)
        exit()

        # file = open(f"data/mydata/txt/000000{i:02d}.txt", 'w+')
        # file.write("CONTOUR\n")
        # for y in range(pmat.shape[0]) :
        #     for x in range(pmat.shape[1]) :
        #         file.write(str(pmat[y][x])); file.write(" ")
        #     file.write("\n")
        
        # i += 1