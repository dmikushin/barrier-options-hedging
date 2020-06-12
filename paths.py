import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#####################################################################
#               Generate random paths for shares
#####################################################################

Spot = 100 # Initial stock price
K = 100 # Strike price of an option
Ts = [0.5, 1, 1.5] # Time to maturity, years
sigma = 0.25 # Volatility
b = 0.1 # Carry rate
r = 0.1 # Yearly interest rate
Hs = [50, 60, 70] # Barrier level

nPaths = 10 # number of paths

# set the random seed https://en.wikipedia.org/wiki/Random_seed
np.random.seed(123)

fig, axs = plt.subplots(len(Hs), len(Ts))

df = []
for iH in range(len(Hs)) :
    for iT in range(len(Ts)) :
        H = Hs[iH]
        T = Ts[iT]

        nSteps = int(252 * T)

        # random generation for the normal distribution with mean 0 and standard deviation 1
        dw = np.random.normal(loc=0.0, scale=1.0, size=nPaths * nSteps)

        # deterministic mode to use for testing
        # f = lambda i, j: (i + j + 2) / (nPaths + nSteps)
        # dw = np.transpose(np.fromfunction(np.vectorize(f), (nPaths, nSteps), dtype=float))

        # reshape normal distribution to 2D
        dw = dw.reshape((nSteps, nPaths))

        S = Spot * np.exp(np.apply_along_axis(np.cumsum, 0, (-0.5 * sigma ** 2) * 1 / nSteps + sigma * np.sqrt(1/nSteps) * dw))

        S = np.concatenate((np.full((1, S.shape[1]), Spot), S), axis = 0)

        payoff = np.maximum(K - S[-1,:], 0) * (np.min(S, 0) < H) # "in" option "put"
        # payoff = np.maximum(S[-1,:] - K, 0) * (np.min(S,0) < H) # "in" option "call"
        # payoff = np.maximum(S[-1,:] - K, 0) * (np.min(S,0) > H) # "out" option "call"
        # payoff = np.maximum(K - S[-1,:], 0) * (np.min(S,0) > H) # "out" option "put"
        
        payoff = payoff * (1 + r) ** (-T)
        price = np.mean(payoff)
        row = pd.Series(data = { 'H' : H, 'T' : T, 'price' : price })
        df.append(row)

        t = range(nSteps + 1)
        for i in range(nPaths) :
            axs[iH, iT].plot(t, S[:,i])
        axs[iH, iT].set_title('H = {:.2f}, T = {:.2f}, price = {:.2f}'.format(H, T, price))

df = pd.concat(df,1).T
print(df)

df.sort_values('H')

plt.subplots_adjust(hspace=0.5)
plt.show()
