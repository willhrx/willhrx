
"""
Monte carlo pricing model

this is a widely used method of option pricing, and something that I have not used a lot since 
my days at uni so i though it might be fun to revisit the concepts and have a play around

@author: willi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random

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

#Plotting for n = 20
plt.bar(x, mc_dice_rolls(n = 20), color = 'dodgerblue')
plt.title('Montecarlo simulation of rolling 2 dice (n = 20)')
plt.xlabel('Sums of the rolls')
plt.ylabel('Probabilities of each outcome')
plt.grid(alpha=0.4, linestyle='--')

#Plotting for n = 100
plt.bar(x, mc_dice_rolls(n = 100), color = 'dodgerblue')
plt.title('Montecarlo simulation of rolling 2 dice (n = 100)')
plt.xlabel('Sums of the rolls')
plt.ylabel('Probabilities of each outcome')
plt.grid(alpha=0.4, linestyle='--')

#Plotting for n = 500
plt.bar(x, mc_dice_rolls(n = 500), color = 'dodgerblue')
plt.title('Montecarlo simulation of rolling 2 dice (n = 500)')
plt.xlabel('Sums of the rolls')
plt.ylabel('Probabilities of each outcome')
plt.grid(alpha=0.4, linestyle='--')

#Plotting for n = 10000
plt.bar(x, mc_dice_rolls(n = 10000), color = 'dodgerblue')
plt.title('Montecarlo simulation of rolling 2 dice (n = 10000)')
plt.xlabel('Sums of the rolls')
plt.ylabel('Probabilities of each outcome')
plt.grid(alpha=0.4, linestyle='--')

