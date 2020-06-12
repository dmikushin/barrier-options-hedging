# Barrier Options Hedging

Assume we want to buy a down-and-in put option with American type of barrier (continuous monitoring) on a single underlying, S<sub>T</sub>. The payoff function at maturity is given by:

<pre>
DIP(T) = max(K - S<sub>T</sub>) * 1{S<sub>t</sub> < H},
</pre>

where H being the barrier level, K is the strike price, S<sub>T</sub> is the underlying price, and T is the maturity. The put gets activated once the spot value S<sub>t</sub> crosses the barrier level, i.e. 1{S<sub>t</sub> < H} = 1.

This position creates some market risk to the portfolio, which we want to hedge. This can be done via either:

* `Dynamic hedging` - replication: replicating the payoff by entering positions into the underlying and borrowing/lending money from a bank account, or
* `Static hedging`: as described in [1] and [2]

## Python pricing function in Python

We create a pricing function for the above barrier option using the Monte Carlo simulation to generate the underlying paths. The underlying paths are following the Geometric Brownian Motion (GBM) "BS" process, as described in Eq. 14 of [2].

## Python pricing function evaluation

The Python pricing function uses the following default presets:

* µ = r = 0.1%
* Volatility σ = 0.25
* Initial stock price S<sub>t0</sub> = 100, K = 100
* Time to expiration T = {0.5, 1.0, 1.5}
* Barrier level H = {50, 60, 70}

