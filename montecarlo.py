
"""
Monte carlo pricing model

this is a widely used method of option pricing, and something that I have not used a lot since 
my days at uni so i though it might be fun to revisit the concepts and have a play around

@author: willi
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
import datetime

"""
My initial memory of monte carlo methods comes initially from using it to compute complex 
probabilities using the law of large numbers. I'll start off with a simple case of rolling 2
dice and adding the numbers together.
"""

#Simulating n rolls and calculating the probabilites of the simulation
def mc_dice_rolls(n = 10):
    x = np.zeros(11)
    for i in range(n):
        roll = random.randint(1, 6) + random.randint(1, 6)
        x[roll - 2] += 1
    return x/n

x = np.arange(2, 13)

fig, axs = plt.subplots(2, 2, figsize = (10, 6))

#Plotting for n = 20
axs[0,0].bar(x, mc_dice_rolls(n = 20), color = 'dodgerblue')
axs[0, 0].set_title('Montecarlo simulation of rolling 2 dice (n = 20)')
axs[0, 0].set_xlabel('Sums of the rolls')
axs[0, 0].set_ylabel('Probabilities of each outcome')
axs[0,0].grid(alpha=0.4, linestyle='--')

#Plotting for n = 100
axs[0,1].bar(x, mc_dice_rolls(n = 100), color = 'deepskyblue')
axs[0, 1].set_title('Montecarlo simulation of rolling 2 dice (n = 100)')
axs[0, 1].set_xlabel('Sums of the rolls')
axs[0, 1].set_ylabel('Probabilities of each outcome')
axs[0,1].grid(alpha=0.4, linestyle='--')

#Plotting for n = 500
axs[1,0].bar(x, mc_dice_rolls(n = 1000), color = 'forestgreen')
axs[1, 0].set_title('Montecarlo simulation of rolling 2 dice (n = 1000)')
axs[1, 0].set_xlabel('Sums of the rolls')
axs[1, 0].set_ylabel('Probabilities of each outcome')
axs[1,0].grid(alpha=0.4, linestyle='--')

#Plotting for n = 10000
axs[1, 1].bar(x, mc_dice_rolls(n = 10000), color = 'mediumaquamarine')
axs[1, 1].set_title('Montecarlo simulation of rolling 2 dice (n = 10000)')
axs[1, 1].set_xlabel('Sums of the rolls')
axs[1, 1].set_ylabel('Probabilities of each outcome')
axs[1, 1].grid(alpha=0.4, linestyle='--')

plt.tight_layout()
plt.show()

print('The expected probability of rolling a 2 is 0.0277777...')
print('The probability of getting a 2 after 20 rolls is ' + str(mc_dice_rolls(n = 20)[0]))
print('The probability of getting a 2 after 100 rolls is ' + str(mc_dice_rolls(n = 100)[0]))
print('The probability of getting a 2 after 500 rolls is ' + str(mc_dice_rolls(n = 500)[0]))
print('The probability of getting a 2 after 10000 rolls is ' + str(mc_dice_rolls(n = 10000)[0]))

"""
As simple as this case is, I feel it really demonstrates the power and potential of the monte
carlo method. As the number of trials rises, both the overall probability distributuion moves 
closer and closer to the expected probability distribution and I feel like I can already see 
how powerful this could be when put to more complex cases. Now let's find see if we can apply
these principals to an option pricing model.


I'll begin with giving a quick explannation of what a european option is. This is a financial
instrument that gives the buyer the option to either buy (call) or sell (put) an underlying asset
for a specified price (strike price) at a specified time in the future (expiration time), not 
earlier, as is allowed with an american-type option.
Our task is to calculate a fair price for this option based on what we know about the nature of
the underlying financial asset.

To begin, I think we should take a look at the price of a stock and watch how it chages over the 
course of a year.
"""

#Downloading ohlcv data on AMZN stock

import yfinance as yf 

apple = yf.Ticker("AAPL")
ohlcv_data = apple.history(period='1y', interval='1d')

prices = ohlcv_data['Close']
dates = ohlcv_data.index

#Plot of the prices

plt.figure(figsize=(10, 6))
plt.plot(dates, prices, color = 'dodgerblue')
plt.title(label = 'Daily AAPL stock prices over the past year')
plt.ylabel(ylabel = 'Prices (USD)')
plt.xlabel(xlabel = 'Dates')
plt.grid(alpha = 0.4)
plt.show()

"""
From looking at this plot, I feel like you get a good sense of two of the main characteristics
that are used to quantifiably model the movement of stock prices: the volatility and the overall
direction.

A common way of modelling the movement of stock prices is with an Ito process, it is very rubust
and will we start off using a version of it to start off with.
"""


#Calculating log change
ohlcv_data['Log Change'] = np.log(ohlcv_data['Close']/ohlcv_data['Close'].shift(-1))
ohlcv_data['Change'] = ohlcv_data['Close'].pct_change()

AAPL_mean = ohlcv_data['Change'].mean()
AAPL_vol = ohlcv_data['Change'].std()

"""
So an Ito process is defined as:
dX = b(X, t)dt + a(X, t)dWt
where X is the process, t is the time increment and Wt is a Wiener process (brownian motion).

A function of this form has many properties and I will not be getting into the details of them
right now, I'll just be starting with a basic form of this process:    

dX = X(1 + mean) dt + t * vol dWt

so X_t+1 is normally distributed with a mean of X_t(1 + AAPL_mean) and standard deviation of
AAPL_vol * X_t
"""

def AAPL_ito(n_0, n):
    y = [n_0]
    for i in range(n -1):
        y.append(y[-1] * (1 + AAPL_mean) + np.random.normal(loc = 0, scale = AAPL_vol * y[-1]))
    return(y)

ito = AAPL_ito(n_0 = prices[0], n = len(prices))

#Plot of the process against the actual prices
plt.figure(figsize=(10, 6))
plt.plot(dates, prices, color = 'dodgerblue')
plt.plot(dates, ito, color = 'dimgrey')
plt.title(label = 'Daily AAPL stock prices over the past year against Ito Process')
plt.ylabel(ylabel = 'Prices (USD)')
plt.xlabel(xlabel = 'Dates')
plt.grid(alpha = 0.4)
plt.show() 

"""
So the idea is to run a large number of simulations of future prices using the stochacstic model
we've made to determine a fair price for an option.
"""

def AAPL_mc(x, k, n):
    price = 0
    for i in range(n):
        AAPL = prices[-1]
        for j in range(x):
            AAPL = AAPL * (1 + AAPL_mean) + np.random.normal(loc = 0, scale = AAPL_vol * AAPL)
        price += max(0, AAPL - k)/n
    return(price)

def blackscholes (X, S, t, v, r):
    d1 = (np.log(S/X) + (r + (v**2/2))*t)/(v*np.sqrt(t))
    d2 = d1 - v*np.sqrt(t)
    C = S*stats.norm.cdf(d1) - X*np.exp(-r*t)*stats.norm.cdf(d2)
    P = X*np.exp(-r*t)*stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1)
    return(C, P)

mc_call = AAPL_mc(100, 200, 100000)
bs_call = blackscholes( 200, S = prices[-1], t = 100/252, v = AAPL_vol, r = 0)[0]

print('Price according to the Black-Scholes pricing model: '+ str(bs_call))
print('Price according to the Monte-Carlo pricing model: '+ str(mc_call))

"""
This is a good start, it seems clear that the monte carlo pricing model values the 200 AAPL call 
with 6 days to expiration quite a bit more than the black scholes model does. There is still quite
bit that can be done to improve our monte carlo pricing model.
Our current function is very computationally intense for large n, and takes a long time as it is
simulating the full path of the stock n times rather than just giving n different outcomes.
"""

sim_dates = [dates[-1] + datetime.timedelta(days=i) for i in range(100)]

#Plot of 100 processes
plt.figure(figsize=(10, 6))
for i in range(100):
    plt.plot(sim_dates, AAPL_ito(n_0 = prices[-1], n = 100 ), color = (random.random(), random.random(), random.random()))
plt.title(label = 'Simulartion of 100 prices 100 days into the future')
plt.ylabel(ylabel = 'Prices (USD)')
plt.xlabel(xlabel = 'Dates')
plt.grid(alpha = 0.4)
plt.show() 

"""
Our pricing model is going through every single point on that graph when all we are interested in
is the last value or the price at expiration, which according to the model, should be centred 
around  
"""




