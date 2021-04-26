import numpy as np

def estimateProjectionMatrix(x, y, P) : 
    Q = np.array([])

    for i in range(10) : 
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
    print(E)
    input("")

    res = np.array([
        [V[11, 0], V[11, 1], V[11, 2],  V[11, 3]],
        [V[11, 4], V[11, 5], V[11, 6],  V[11, 7]],
        [V[11, 8], V[11, 9], V[11, 10], V[11, 11]]
    ])

    return res 

if __name__ == "__main__" : 
    file = open("data/mydata/txt/calibration.txt", 'r')
    lines = file.readlines()
    i_num = int(lines.pop(0))

    P = []
    X = np.empty(i_num, dtype=list)
    Y = np.empty(i_num, dtype=list)
    i = 0
    for i in range(i_num) :
        p_num = int(lines.pop(0))
        x = []
        y = []
        p = np.array([])
        for j in range(p_num) :
            line = lines.pop(0)
            words = line.split()
            pt = np.array([float(words[0]), float(words[1]), float(words[2]), float(words[3])])
            x.append(int(words[4]))
            y.append(int(words[5]))
            if np.array_equal(p, []) :
                p = pt 
            else : 
                p = np.vstack((p, pt))
        p = p.T
        X[i] = x 
        Y[i] = y
        P.append(p)
        i += 1
    file.close()
    for i in range(len(X)) : 
        pmat = estimateProjectionMatrix(X[i], Y[i], P[i])
        # file = open(f"data/mydata/txt/000000{i:02d}.txt", 'w+')
        # file.write("CONTOUR\n")
        # for y in range(pmat.shape[0]) :
        #     for x in range(pmat.shape[1]) :
        #         file.write(str(pmat[y][x])); file.write(" ")
        #     file.write("\n")