import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from scipy import optimize

#####################################################################
#               Generate random paths for shares
#####################################################################

Spot = 100  # Initial stock price
K = 100  # Strike price of an option
Ts = [0.5, 1, 1.5]  # Time to maturity, years
sigma = 0.25  # Volatility
r = 0.01  # Yearly interest rate
mu = 0.01
Hs = [50, 60, 70]  # Barrier level

# number of paths

# set the random seed https://en.wikipedia.org/wiki/Random_seed
np.random.seed(123)

##################
# exercise 1
##################

# nSteps=300; Spot=100; H=70; nPaths = 1000*10

# create a pricing function
def gen_path(Spot, nPaths, nSteps):
    dw = np.random.normal(loc=0.0, scale=1.0, size=(nSteps, nPaths))
    S = Spot * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) * 1 / 252 + sigma * np.sqrt(1 / 252) * dw, axis=0))
    return S


def barrier_price(nSteps, Spot, H=80, nPaths=1000 * 10):
    T = nSteps / 252
    # random generation for the normal distribution with mean 0 and standard deviation 1
    S = gen_path(Spot, nPaths, nSteps)
    payoff = np.maximum(K - S[-1, :], 0) * (np.min(S, 0) < H)  # "in" option "put"
    payoff = payoff * (1 + r) ** (-T)
    price = np.mean(payoff)
    return price


##################
# exercise 2
##################
df = []
for iH in range(len(Hs)):
    for iT in range(len(Ts)):
        H = Hs[iH]
        nSteps = int(252 * Ts[iT])

        price = barrier_price(nSteps, 100, H)
        row = pd.Series(data={'H': H, 'T': Ts[iT], 'price': price})
        df.append(row)

df = pd.concat(df, 1).T
print(df)

