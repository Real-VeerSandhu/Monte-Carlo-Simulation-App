## Monte Carlo Stock Portfolio Simulation

### Overview
This project is a **Monte Carlo Stock Portfolio Simulation** web app built using **Streamlit**. It allows users to select a portfolio of stocks, allocate percentage weights, and run Monte Carlo simulations to forecast future portfolio performance based on historical stock data.

### Features
- **Interactive stock selection**: Choose from a list of stocks.
- **Customizable allocations**: Assign percentage allocations to each stock.
- **Historical data retrieval**: Fetches stock closing prices using Yahoo Finance.
- **Monte Carlo simulation**: Simulates multiple future scenarios based on past stock returns.
- **Flexible parameters**:
  - Portfolio start date
  - Number of simulations
  - Time range for simulations
  - Initial portfolio value
- **Graphical visualization**:
  - Matplotlib for static plots
  - Plotly for interactive visualizations

### Installation
Ensure you have Python installed, then install dependencies:

```
pip install streamlit pandas numpy matplotlib yfinance plotly
```

### Usage
1. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```
2. Select stocks and allocate percentages (must sum to 100%).
3. Adjust simulation parameters.
4. Click **Start Simulation** to generate Monte Carlo simulations.
5. View the results as static or interactive graphs.

### How It Works
1. **Fetches historical stock data**: Uses Yahoo Finance to get closing prices.
2. **Calculates returns and covariance matrix**: Computes daily percentage changes and the covariance between stocks.
3. **Monte Carlo Simulation**:
   - Generates random future returns based on a multivariate normal distribution.
   - Applies Cholesky decomposition to model correlations between stocks.
   - Simulates multiple portfolio performance paths.
4. **Visualization**: Displays simulations using either Matplotlib or Plotly.

### Dependencies
- **Streamlit**: For the interactive UI
- **Pandas**: Data handling
- **NumPy**: Numerical operations
- **Matplotlib**: Static visualization
- **Plotly**: Interactive visualization
- **Yahoo Finance (yfinance)**: Stock market data retrieval

### Example Output
![Example Monte Carlo Simulation](https://via.placeholder.com/800x400?text=Example+Monte+Carlo+Plot)

### License
This project is licensed under the MIT License.

### Acknowledgments
Built with using Python, Streamlit, Numpy, Pandas, Yahoo Finance, and Plotly.