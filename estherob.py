import math
from scipy.stats.distributions import norm
import matplotlib.pyplot as plt

error = True
message = "This input is incorrect. Please enter p for put or c for call"
while error:
    error = False
    putCall = input("Do you want to price a put(p) or a call(c) option? " )
    if putCall == 'p':
        phi = -1
    elif putCall == 'c':
        phi = 1
    else:
        print(message)
        error = True

S = float(input("Please enter the current Stock price- "))
X = float(input("Please enter the strike price of the option- "))
T = float(input("Please enter the time to maturity(in days)- "))
T = float(T)/365

sigma = float(input("Please enter the volatility of the underlying asset- "))

b = float(input("Please enter the carry rate- "))
r = float(input("Please enter the yearly interest rate- "))
H = float(input("Please enter the barrier height- " ))
K = float(input("How high is the rebate you receive if you get knocked out?- "))

message3 ="Please enter out or in."

error3 = True
while error3:
    inOut = input("Please type whether it is an out or an in barrier.(out/in)- ")
    if inOut == 'out' or inOut == 'in':
        error3 = False
    else:
        print(message3)

X = float(X)
b = float(b)
H = float(H)
K = float(K)
S = float(S)

# Function for evaluating the option price given the underlying price
def price_option(S) :
    if putCall == 'c':
        phi = 1
    else:
        phi = -1

    if S>H:
        eta = 1
    else:
        eta = -1

    sigma2 = sigma**2
    mu = (b - (sigma2 / 2)) / sigma2
    landa = math.sqrt(mu ** 2.0 + (2 * r) / sigma2)
    x1 = (math.log(S / X)) / (sigma * math.sqrt(T)) + (1 + mu) * sigma * math.sqrt(T)
    x2 = (math.log(S / H)) / (sigma * math.sqrt(T)) + (1 + mu) * sigma * math.sqrt(T)
    y1 = math.log((H ** 2) / (S * X)) / (sigma * math.sqrt(T)) + (1 + mu) * sigma * math.sqrt(T)
    y2 = math.log(H / S) / (sigma * math.sqrt(T)) + (1 + mu) * sigma * math.sqrt(T)
    z = math.log(H / S) / (sigma * math.sqrt(T)) + landa * sigma * math.sqrt(T)

    n1 = phi * x1 - phi * sigma * math.sqrt(T)
    A = phi * S * math.exp(b - r) * norm.cdf(phi * x1, 0, 1) - phi * X * math.exp(-r * T) * norm.cdf(n1, 0, 1)

    n2 = phi * x2 - phi * sigma * math.sqrt(T)
    B = phi * S * math.exp(b - r) * norm.cdf(phi * x2, 0, 1) - phi * X * math.exp(-r * T) * norm.cdf(n2, 0, 1)

    n3 = eta * y1
    n4 = eta * y1 - eta * sigma * math.sqrt(T)
    C = phi * S * math.exp(b - r) * (H / S) ** (2 * (mu + 1)) * norm.cdf(n3, 0, 1) - \
        phi * X * math.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(n4, 0, 1)

    n5 = eta * y2
    n6 = eta * y2 - eta * sigma * math.sqrt(T)
    D = phi * S * math.exp(b - r) * (H / S) ** (2 * (mu + 1)) * norm.cdf(n5,0,1) - \
        phi * X * math.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(n6, 0, 1)

    n9 = norm.cdf(eta * x2 - eta * sigma * math.sqrt(T), 0, 1)
    n10 = norm.cdf(eta * y2 - eta * sigma * math.sqrt(T), 0, 1)
    E = K * math.exp(-r * T) * (n9 - (H / S) ** (2 * mu) * n10)

    n11 = norm.cdf(z * eta, 0,1)
    n12 = norm.cdf(eta * z - 2 * eta * landa * sigma * math.sqrt(T), 0, 1)
    F = K * math.exp(-r * T) * ((H / S) ** (mu + landa) * (n11) - (H / S) ** (mu - landa) * n12)

    if putCall == 'p':
        if X < H:
            if S > H:
                if inOut == 'in':
                    price = A + E
                elif inOut == 'out':
                    price = F
            else:
                if inOut == 'in':
                    price = C + E
                elif inOut == 'out':
                    price = A - C + F
        else:
            if S > H:
                if inOut == 'in':
                    price = B - C + D + E
                elif inOut == 'out':
                    price = A - B + C - D + F
            else:
                if inOut == 'in':
                    price = A - B + D + E
                elif inOut == 'out':
                    price = B - D + F
    else:
        if X < H:
            if S > H:
                if inOut == 'in':
                    price = A - B + D + E
                elif inOut == 'out':
                    price = B - D + F
            else:
                if inOut == 'in':
                    price = B - C + D + E
                elif inOut == 'out':
                    price = A - B + C - D + F
        else:
            if S > H:
                if inOut == 'in':
                    price = C + E
                elif inOut == 'out':
                    price = A - C + F
            else:
                if inOut == 'in':
                    price = A + E
                elif inOut == 'out':
                    price = F

    if price < 0:
        price = 0

    return price

# Underlying price
print(price_option(S))
S = int(S)
u_price = range (S - 25, S + 50)

# Option price
op_price = []
for i in range(len(u_price)):
    op_price.append(price_option(u_price[i]))

print(u_price)
print(op_price)

# Plotting the underlying price vs. the option price
plt.plot(u_price, op_price, color = 'red')
plt.xlabel('Underlying price', fontsize = 14)
plt.ylabel('Option price', fontsize = 14)
plt.title('PUT Out Barrier X < H', fontsize = 20)
plt.show()

