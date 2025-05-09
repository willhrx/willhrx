# -*- coding: utf-8 -*-
"""
Portfolio optimization sim

@author: willi
"""

import yfinance as yf
import numpy as np
import pandas as pd

tickers = ['AAPL', 'NVDA', 'META', 'MSFT', 'GOOG']
test_changes = {'AAPL' : pd.Series([-1, 0.5, 0.2]), 'NVDA' : pd.Series([0.2, 1, 0.2]),
                'META' : pd.Series([1, 2, -1]), 'MSFT' : pd.Series([0.4, -0.3, 1]),
                'GOOG' : pd.Series([-0.7, -1, 1.1])}

ohlcv_dict = {}

for ticker in tickers:
    df = yf.download(tickers = ticker, period = '1y', interval = '1d')
    df['Change'] = df['Close'].pct_change()
    ohlcv_dict[ticker] = df

change_mat = [ohlcv_dict[ticker]['Change'] for ticker in tickers]
test_mat = [test_changes[ticker] for ticker in tickers]  
cov_mat = np.cov(test_mat)

for i in range(len(tickers)):
    for j in range(len(tickers)):
        if i != j:
            print('The covariance between ' + str(tickers[i]) + ' and ' + str(tickers[j]) + ' is ' + str(cov_mat[i, j]))