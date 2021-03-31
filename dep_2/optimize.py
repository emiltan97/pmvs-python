from scipy.optimize import minimize
from numpy.random import rand
    
def objective(x):
    return x[0]**2.0 + x[1]**2.0

# Objective function 
# Set xo
# Set lower bound 
# Set upper bound 
# Set max eval 

if __name__ == "__main__" : 
    # define range for input
    r_min, r_max = -5.0, 5.0
    # define the starting point as a random sample from the domain
    pt = r_min + rand(2) * (r_max - r_min)
    # perform the search
    result = minimize(objective, pt, method='nelder-mead')
    # summarize the result
    print('Status : %s' % result['message'])
    print('Total Evaluations: %d' % result['nfev'])
    # evaluate solution
    solution = result['x']
    evaluation = objective(solution)
    print('Solution: f(%s) = %.5f' % (solution, evaluation))