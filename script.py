import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import pybobyqa
from scipy import optimize
import seaborn as sns
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


##################
# exercise 3 + 4
##################

# I didn't understand how to numercially implement the equation in the paper, instead I propose a numerical approximation of the static replicaiton protfolio

# this function estimates the price
def barrier_price(nSteps=126, Spot=100, H=80, nPaths=1000 * 10, ret_='payoff', rep_w=None):
    T = nSteps / 252
    # random generation for the normal distribution with mean 0 and standard deviation 1
    S = gen_path(Spot, nPaths, nSteps)
    barrier_payoff = np.maximum(K - S[-1, :], 0) * (np.min(S, 0) < H)  # "in" option "put"

    # generate a bunch of call and put options with strike between MIN_K and MAX_X
    MIN_K = 0
    MAX_K = 160
    rep_mat = []
    init_x = []
    KK = []
    for k in np.arange(MIN_K, MAX_K, 1):
        init_x.append(0)
        init_x.append(0)
        rep_mat.append(np.maximum(k - S[-1, :], 0))
        rep_mat.append(np.maximum(S[-1, :] - k, 0))
        KK.append(k)
        KK.append(k)
    rep_mat = np.array(rep_mat)
    init_x = np.array(init_x)

    if ret_ == 'payoff':
        return barrier_payoff, rep_mat, init_x
    if ret_ == 'price':
        rep_price = np.mean(rep_mat, 1) * (1 + r) ** (-T)
        bar_price = np.mean(barrier_payoff) * (1 + r) ** (-T)
        return bar_price, rep_price
    if ret_ == 'port':
        hedge = (np.mean(rep_mat, 1) * (1 + r) ** (-T)) @ rep_w
        real = np.mean(barrier_payoff) * (1 + r) ** (-T)
        return real, hedge


# use the function to generate a bunch of final payoff
barrier_payoff, rep_mat, init_x = barrier_price(nSteps=126, Spot=100, H=80, nPaths=1000 * 10, ret_='payoff')

# create a cost function with the mean square replication error of payoff
def func(x):
    rep = x.reshape(1, -1) @ rep_mat
    return np.mean(np.square(barrier_payoff.reshape(1, -1) - rep))


# set the bounds so each individual
bnds = [[-1, 1] for x in init_x.tolist()]
#solve the optimisation
opt = optimize.minimize(func, init_x, bounds=bnds)
# extract the optimal weights of the replicating portfolio
w = opt['x']
print('Replication error with no hedge portfolio', func(init_x))
print('Replication error on payoff', func(w))

# we create some path (only 10 because it's computationally heavy)
real_mat = []
hedge_mat = []
s_path = gen_path(100, nPaths=10, nSteps=126)
k = 0
for i in range(s_path.shape[0]):
    real_vec = []
    hedge_vec = []
    for j in range(s_path.shape[1]):
        # for each day we compute the price of the barrier option and the static replication protfolio's value
        real, hedge = barrier_price(nSteps=126 - i, Spot=s_path[i, j], H=80, nPaths=1000 * 10, ret_='port', rep_w=w)
        real_vec.append(real)
        hedge_vec.append(hedge)
        k += 1
        if (k % 25 == 0):
            print(k, '/', s_path.shape[0] * s_path.shape[1])
    real_mat.append(real_vec)
    hedge_mat.append(hedge_vec)
real_mat = np.array(real_mat)
hedge_mat = np.array(hedge_mat)

# compute the rep error.
rep_error = np.abs(real_mat - hedge_mat)

# plot the distribution of the error
sns.distplot(rep_error.flatten())
plt.title('replication error')
plt.show()

# plot the distribution fo the portfolios
sns.distplot(real_mat.flatten(), label='Real portfolio')
sns.distplot(hedge_mat.flatten(), label='Replicating portfolio')
plt.legend()
plt.show()

##################
# Exercise 5
##################

# The numerical static approach can easily be extended to attempt a static hedging strategy.
# The universe of call and put options needs to be extended to include options on all 3 assets in the baskets.
# The rest of the optimization procedure can stay the same.
#
# However, it is clear that the computaitonal cost will grow exponentially and better optimization procedure may be require.
#
# It's also obvious that dynamic rebalancing may strongly improve the replication quality of such structured products.
# Indeed, if assets 1's values goes through the roof, while asset 2 goes close to zero, the probability that asset 1 will ever be relevant to the payoff DIP(T) becomes negligable.
# It follows that options on asset 1 will not be usefull in replicating the DIP's values.
#
# A simple approach could be to estimate the static portfolio on each individual asset for a single barrier option.
# The hedging portoflio of the worst basket of options could be some weighted sum of those three payoffs.
# The weights in this sum would be function of the respective spot value of the three assets and their volatility---that is, the probabiltiy that asset i's values will be the smallest one at maturity.
#
