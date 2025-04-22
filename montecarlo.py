
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
axs[1,0].bar(x, mc_dice_rolls(n = 500), color = 'forestgreen')
axs[1, 0].set_title('Montecarlo simulation of rolling 2 dice (n = 500)')
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
how powerful this could be when put to more complex cases.
"""













