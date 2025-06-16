"""
MC Stock Portfolio Simulation
"""

import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime as dt
import yfinance as yf

import plotly.graph_objs as go



st.set_page_config(
    page_title="Monte Carlo Stock Portfolio Simulatione",
    page_icon="beta.svg",  
    layout="wide",  
)
st.title('Monte Carlo Stock Portfolio Simulation')

col1, col2 = st.columns([1,2])

df = pd.read_csv('stock_info.csv')
tickers = df['Ticker'].tolist()



def get_data(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()

    if isinstance(stocks, str):
        covMatrix = pd.DataFrame({stocks: returns.var()}, index=[stocks])
    elif len(stocks) == 1:
        stock = stocks[0]
        covMatrix = pd.DataFrame({stock: returns.var()}, index=[stock])
    else:
        # For multiple stocks
        covMatrix = returns.cov()
    
    return meanReturns, covMatrix


validBalance = False
with col1:


    options = st.multiselect(
        "Select Stocks",
        tickers
    )

    st.write("Portfolio:", options)

    if options:

        # Dictionary to store allocations
        allocations = {}
        total_allocation = 0

        for ticker in options:
            allocation = st.number_input(
                f"Percentage allocation for {ticker}:",
                min_value=0,
                max_value=100,
                value=0,
                step=1,
            )
            allocations[ticker] = allocation
            total_allocation += allocation

        if total_allocation != 100:
            st.warning(f"Total allocation is {total_allocation}%. It should be 100%.")
        else:
            st.success("Total allocation is 100%. Your portfolio is balanced.")
            validBalance = True

        daysAgo = st.slider('Portfolio Start Date (Days)', min_value=100,
                max_value=500, value=250, step=1)

        endDate = dt.datetime.now()
        startDate = endDate - dt.timedelta(days=daysAgo)
        
        mc_sims = st.slider('MC Simulation Count', min_value=5,
                max_value=1000, value=250, step=1)
        
        T = st.slider('Simulation Time Range (Days)', min_value=20,
                max_value=1000, value=100, step=1)
        
        initialPortfolio = st.slider('Initial Portfolio Value ($)', min_value=1000,
                max_value=100000, value=100, step=1)

        on = st.toggle("Plt Graph")

    else:
        st.write("Please select at least one stock.")
        
with col2:
    if validBalance and st.button('Start Simulation'):
        meanReturns, covMatrix = get_data(options, startDate, endDate)

        numpy_array = np.array(list(allocations.values()))
        weights = numpy_array / 100.0
        
        
        meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
        meanM = meanM.T

        portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)


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


        if on:
            fig, ax = plt.subplots()
            for i in range(portfolio_sims.shape[1]):
                ax.plot(portfolio_sims[:, i], label=f'Simulation_{i+1}')

            # Set labels and title
            ax.set_xlabel('Days')
            ax.set_ylabel('Value')
            ax.set_title('Monte Carlo Simulations')

            # Show plot
            st.pyplot(fig)
        else:

            fig = go.Figure()

            # Add each line to the plot
            for i in range(portfolio_sims.shape[1]):
                fig.add_trace(go.Scatter(x=np.arange(portfolio_sims.shape[0]), y=portfolio_sims[:, i], mode='lines', name=f'Simulation_{i}'))

            # Set x-axis and y-axis labels
            fig.update_layout(xaxis_title='Days', yaxis_title='Value')

            # Show the plot
            st.plotly_chart(fig)
       
        st.write(portfolio_sims)

st.caption('Built with Python, Streamlit, Numpy, Pandas, Yahoo Finance, Plotly')
st.caption('Veer Sandhu - 2024')
st.caption("[Github](https://github.com/Real-VeerSandhu/Monte-Carlo-Simulation-App)")

