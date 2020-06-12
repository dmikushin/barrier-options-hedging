# Based on the code by Mwangota Lutufyo and Omotesho Latifat Oyinkansola
# http://janroman.dhis.org/stud/I2016/Barriar/Barriar.pdf

import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from numpy.lib.scimath import sqrt, log
from scipy import stats

def bs_call(S, X, T, rf, sigma) :
    """
    Black-Scholes-Merton option model call
    S: current stock price
    X: exercise price
    T: maturity date in years
    rf: risk-free rate (continusouly compounded)
    sigma: volatility of underlying security
    """
    d1 = (log(S / X) + (rf + sigma * sigma / 2.) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * stats.norm.cdf(d1) - X * exp(-rf * T) * stats.norm.cdf(d2)

def bs_put(S, X, T, rf, sigma) :
    """
    Black-Scholes-Merton option model put
    S: current stock price
    X: exercise price
    T: maturity date in years
    rf: risk-free rate (continusouly compounded)
    sigma: volatility of underlying security
    """ 
    d1 = (log(S / X) + (rf + sigma * sigma /2.) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return - S * stats.norm.cdf(-d1) + X * exp(-rf * T) * stats.norm.cdf(-d2)

def cnd(X) :
    """ Cumulative standard normal distribution
    cnd(x): x is a scale
    e.g.,
    >>> cnd(0)
    0.5000000005248086
    """
    (a1,a2,a3,a4,a5) = (0.31938153, -0.356563782, 1.781477937, -1.821255978, 1.330274429)
    L = abs(X)
    K = 1.0 / (1.0 + 0.2316419 * L)
    w = 1.0 - 1.0 / sqrt(2 * pi) * exp(-L * L/2.) * (a1 * K + a2 * K * K + a3 * pow(K, 3) + a4 * pow(K,4) + a5 * pow(K,5))
    if X < 0 :
        w = 1.0 - w
    return w

if __name__ == "__main__" :
    np.random.seed(123)
    S0 = 60
    x = 60
    barrier = 61
    T = 0.5
    n_steps = 30
    r = 0.05
    sigma = 0.2
    n_simulation = 5
    dt = T / n_steps

    S = np.zeros([n_steps], dtype = float)
    time_ = range(0, n_steps, 1)

    c = bs_call(S0, x, T, r, sigma)
    
    outTotal = 0.
    inTotal = 0.
    n_out = 0
    n_in = 0

    for j in range(0, n_simulation) :
        S[0] = S0
        inStatus = False
        outStatus = True
        for i in time_[:-1] :
            e = np.random.normal()
            S[i+1] = S[i] * exp((r - 0.5 * pow(sigma, 2)) * dt + sigma * sqrt(dt) * e)
            if S[i+1] > barrier :
                outStatus = False
                inStatus = True
            plt.plot(time_, S)
            if outStatus == True :
                outTotal += c
                n_out += 1
            else :
                inTotal += c
                n_in += 1
            S = np.zeros(int(n_steps)) + barrier
            plt.plot(time_, S, '.-')
            upOutCall = round(outTotal / n_simulation, 3)
            upInCall = round(inTotal / n_simulation, 3)
            plt.figtext(0.15, 0.8, 'S = {}, X = {}'.format(S0, x))
            plt.figtext(0.15, 0.76, 'T = {}, r = {}, sigma = {}'.format(T, r, sigma))
            plt.figtext(0.15, 0.6, 'barrier = {}'.format(barrier))
            plt.figtext(0.40, 0.86, 'call price = {}'.format(round(c,3)))
            plt.figtext(0.40, 0.83, 'up_and_out_call=' + str(upOutCall) + '=' + str(n_out) + '/' + str(n_simulation) + '*' + str(round(c,3)) + ')')
            plt.figtext(0.40, 0.80, 'up_and_in_call =' + str(upInCall) + '(=' + str(n_in) + '/' + str(n_simulation))
            plt.title('Up-and-out and up-and-in parity (# of simulations = %d ' % n_simulation + ')')
            plt.xlabel('Total number of steps =' + str(int(n_steps)))
            plt.ylabel('stock price')
            plt.show()

