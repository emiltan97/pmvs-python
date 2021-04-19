from math import cos
import numpy as np 
from scipy.optimize import minimize

iter = 1

def rosen(x) : 
    # return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    # return (1-(x[0] + x[1]))**2
    return 1

def callback(x) :
    global iter
    print(f'{iter:4d}    {x[0]:8f}    {x[1]:8f}') 
    iter += 1

x0 = np.array([5, 2])
print(f'{"Iter"}    {"X1":8s}    {"X2":8s}')
res = minimize(rosen, x0, method='nelder-mead', options={'xatol':1e-8, 'disp':True}, callback=callback)
print(res.x)