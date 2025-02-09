"""
Implement the Monte Carlo Method to simulate a stock portfolio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

def get_data(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()

    return meanReturns, covMatrix

stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks = [stock + '.AX' for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300) # very important parameters

meanReturns, covMatrix = get_data(stocks, startDate, endDate)

weights = np.random.random(len(meanReturns)) # get random num from 0 to 1

weights /= np.sum(weights)

print(weights)

# Monte Carlo Simulation
# number of simulations

mc_sims = 100
T = 100 # timeframe in days

meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

initialPortfolio = 10000

# Assume daily returns are distributed by a Multivariate Normal Distribution
# Use Cholesky Decomposition to determine Lower Triangular Matrix
for m in range(0, mc_sims):
    # MC loops
    Z = np.random.normal(size=(T, len(weights))) # T by weights (number of stocks)

    # lower triangular matrix
    L = np.linalg.cholesky(covMatrix) # number stocks by number of stocks

    # inner product = dot product
    dailyReturns = meanM + np.inner(L, Z)
    portfolio_sims[:, m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio

plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('Monte Carlo Stock Portfolio Simulation')
plt.show()