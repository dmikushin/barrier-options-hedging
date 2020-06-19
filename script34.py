import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from scipy import optimize

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
plt.hist(rep_error.flatten(), bins=50)
plt.title('replication error')
plt.show()

# plot the distribution fo the portfolios
plt.hist(real_mat.flatten(), bins=50, label='Real portfolio')
plt.hist(hedge_mat.flatten(), bins=50, label='Replicating portfolio', alpha=0.5)
plt.legend()
plt.show()
