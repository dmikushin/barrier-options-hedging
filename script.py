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

# deterministic mode to use for testing
# f = lambda i, j: (i + j + 2) / (nPaths + nSteps)
# dw = np.transpose(np.fromfunction(np.vectorize(f), (nPaths, nSteps), dtype=float))

# reshape normal distribution to 2D
dw = dw.reshape((nSteps, nPaths))

Spot = 13310
vola = 0.2

S = Spot * np.exp(np.apply_along_axis(np.cumsum, 0, (-0.5*vola**2)*1/nSteps + vola*np.sqrt(1/nSteps)*dw))

S<-rbind(rep(Spot,ncol(S)),S)

