# -*- coding: utf-8 -*-
"""
Commodity volatility analysis

Aim: To analyse the implied volatility term structure of commodities and explore
ways to accuratley model such financial instruments using a range of different methods.

@author: willi
"""
#Let's start with the Volious python libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from arch import arch_model


"""
I have finally got my hands on some data that I am very happy to work  with. Historical data
on US Wheat prices from 2001 - 2022
"""

total_data = pd.read_csv("C:/Users/willi/OneDrive/Documents/Data/commodity 2000-2022.csv")
wheat_data = total_data[total_data['Symbol'] == 'US Wheat']
#Changing the date inputs from the origional strings to datetime 
wheat_data['Date'] = pd.to_datetime(wheat_data['Date'])
wheat_data.set_index('Date', inplace=True)
wheat_data = wheat_data.drop(columns = ['Symbol'])
wheat_data['Change'] = wheat_data['Close'].pct_change()*100

"""
Volatility can be calculated in a number of ways so I thought it best to define the method I'll
be using to define volatility.
I will be using calculating anualised 30 day rolling volatility using the sample standard
deviation using a zero avergae assumption:
    (sum i = 1 - 30 (xi)**2)/29
    defining xi as the logarithmic change
"""
wheat_data['Log Change'] = np.log(wheat_data['Close']/wheat_data['Close'].shift(1))
wheat_data['30 Day Rolling Vol'] = wheat_data['Log Change'].rolling(window = 30).apply(
    lambda x: np.sqrt(np.sum(x**2)/29), raw=True
)*np.sqrt(len(wheat_data.loc['2012-01-01':'2016-12-31'])/5)

#Adding exponentially wieghted moving averages of the volatility for later use
wheat_data['com 100'] = wheat_data['30 Day Rolling Vol'].ewm(com = 100, min_periods = 150).mean()

# Aggregating data into yearly averages
yearly_data = wheat_data.resample('Y').mean()
yearly_data.index = pd.to_datetime(yearly_data.index)
yearly_data['Change'] = yearly_data['Close'].pct_change()


#Plotting the yearly price
plt.figure(figsize=(10, 6))
plt.plot(yearly_data.index.year, yearly_data['Close'], marker='o', linestyle='-', color='darkorange', linewidth=2)
plt.title('Yearly Average Price of US Wheat (2001–2022)', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average Price (USD)', fontsize=14)
plt.xticks(yearly_data.index.year, rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(alpha=0.4, linestyle='--')
plt.tight_layout()

#Plotting the 30 day volatility with the year by year change in average price 
plt.figure(figsize=(10, 6))
start = pd.Timestamp('2001-12-31 00:00:00')
end = pd.Timestamp('2020-12-31 00:00:00')
plt.plot(yearly_data.index, yearly_data['Change'], marker='o', linestyle='-', color='steelblue', alpha = 0.5, linewidth=2, label = 'Yearly Change')
plt.axhline(y=0, color='tab:pink', linestyle='-', label='Zero Mean Assumption', alpha = 0.6)
plt.plot(wheat_data.index, wheat_data['30 Day Rolling Vol'], label = '30 Day Rolling Vol',  color = 'tab:cyan')
plt.title('30 Day Rolling Volatility and Yearly Average ', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=14)
plt.xticks(ticks=yearly_data.index, labels=yearly_data.index.year, rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(start, end)
plt.grid(alpha=0.4, linestyle='--')
plt.legend()
plt.tight_layout()

"""
Let's take a closer look at the volatility in the period of 2012-2017 to see if we can
accuratly model this. Volatility is often said to be mean reverting so taking a look at the
mean along the plot may be interesting.
"""

sample_start = '2012-01-01'
sample_end = '2016-12-31'
wheat_sample = wheat_data.loc[sample_start : sample_end]

#A quick look at the statistical summary
wheat_sample['30 Day Rolling Vol'].describe()

#Plotting the volatility in the smaller window
plt.figure(figsize = (10, 6))
plt.plot(wheat_sample.index, wheat_sample['30 Day Rolling Vol'], color = 'tab:purple', label = 'Rolling Volatility')
plt.plot(wheat_sample.index, wheat_sample['com 100'], color = 'steelblue', label = 'EWMA')
plt.axhline(y=wheat_sample['30 Day Rolling Vol'].mean(), color='tab:pink', linestyle='-', label='Sample Mean', alpha = 0.6)
plt.axhline(y=wheat_data['30 Day Rolling Vol'].mean(), color='lightseagreen', linestyle='-', label='"Historical" Mean', alpha = 0.6)
plt.title('Volatility of wheat recorded over a 5 year window', fontweight = 'bold')
plt.yticks(fontsize=12)
plt.grid(alpha=0.4, linestyle='--')
plt.legend()
plt.tight_layout()

"""
It appears to be the case that the volatility is mean reverting when looking at the sample mean
but it likely makes sense to use a moving average. I will be adding an exponential moving average
to the data.
"""

#Overlaying EWMA on the volatility plot
plt.figure(figsize = (10, 6))
plt.plot(wheat_sample.index, wheat_sample['com 100'], color = 'steelblue', label = 'EWMA')
plt.plot(wheat_sample.index, wheat_sample['30 Day Rolling Vol'], color = 'tab:purple', label = 'Rolling Volatility')
plt.title('Volatility of wheat recorded over a 5 year window with EWMA', fontweight = 'bold')
plt.yticks(fontsize=12)
plt.grid(alpha=0.4, linestyle='--')
plt.legend()
plt.tight_layout()

"""
I dont like the looks of this, it just appears to be lagging behind the actual trend of
the volatility, being a little too late to the party. Clearly it's a start and smooths out the 
graph while keeping the general trend of the data but I think we could do better.
"""

#Making an ARIMA model for the volatility
arima1 = ARIMA(wheat_sample['30 Day Rolling Vol'], order = (1, 0, 0)) #Missing dates and frequency not specified at this time, in order to fix I need to set a frequency and fill days the prices are recorded
arima2 = ARIMA(wheat_sample['30 Day Rolling Vol'], order = (2, 0, 2))
arima3 = ARIMA(wheat_sample['30 Day Rolling Vol'], order = (0, 0, 1))
#Plotting ARIMA against vol
plt.figure(figsize = (10, 6))
plt.plot(wheat_sample.index, wheat_sample['30 Day Rolling Vol'], color = 'midnightblue', label = 'Rolling Volatility')
plt.plot(arima1.fit().fittedvalues, color = 'mediumturquoise', label = 'ARIMA 1', alpha = 0.6)
plt.plot(arima2.fit().fittedvalues, color = 'springgreen', label = 'ARIMA 2',  alpha = 0.6)
plt.plot(arima3.fit().fittedvalues, color = 'salmon', label = 'ARIMA 3',  alpha = 0.6)
plt.title('Volatility of wheat recorded over a 5 year window with ARIMA', fontweight = 'bold')
plt.yticks(fontsize=12)
plt.grid(alpha=0.4, linestyle='--')
plt.legend()
plt.tight_layout()

"""
The ARIMA model appears to follow the rolling volatility extremely closely, the two lines seem
indistingushable.
"""
#Inspecting MAE of the 3 differnt models
print('The mean absolute error of  ARIMA1 is ' + str(mean_absolute_error(wheat_sample['30 Day Rolling Vol'], arima1.fit().fittedvalues)))
print('The mean absolute error of  ARIMA2 is ' + str(mean_absolute_error(wheat_sample['30 Day Rolling Vol'], arima2.fit().fittedvalues)))
print('The mean absolute error of  ARIMA3 is ' + str(mean_absolute_error(wheat_sample['30 Day Rolling Vol'], arima3.fit().fittedvalues)))

"""
ARIMA 1 is has the lowwest MAE of the 3 models, that doesnt mean that it will nescessarily 
be the best for future predictions
"""














    











