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
