import numpy as np

#####################################################################
#               Generate random paths for shares
#####################################################################

nPaths = 10 # number of paths
nSteps = 252 # number of days

# set the random seed https://en.wikipedia.org/wiki/Random_seed
np.random.seed(123)

# random generation for the normal distribution with mean 0 and standard deviation 1
dw = np.random.normal(loc=0.0, scale=1.0, size=nPaths*nSteps)

# reshape normal distribution to 2D
dw = dw.reshape((nPaths, nSteps))

Spot = 13310
vola = 0.2
