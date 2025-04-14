
# 📈 Enhanced Stock Forecasting Streamlit App 📉

## 🚀 Project Overview

The **Enhanced Stock Forecasting Streamlit App** is an **interactive web application** designed for stock price forecasting, analysis, and portfolio simulation. The app empowers users with machine learning models, including **AutoReg**, **Prophet**, and advanced financial indicators, to forecast stock trends and make informed investment decisions.

Built with **Streamlit**, **Prophet**, **AutoReg**, **Plotly**, and **Scikit-learn**, this app combines powerful forecasting tools with interactive visualizations, making it a valuable tool for financial analysts and investors.

---

## 🔑 Key Features

### 📊 Interactive Stock Price Visualization
- Visualize **historical stock prices** with **line plots** and **candlestick charts** using **Plotly**.
- **Real-time interactivity** for zooming and selecting time periods for price analysis.

### 🔮 Stock Price Forecasting
- **AutoReg** and **Prophet** models to forecast stock prices up to **120 days**.
- **Automatic model selection** based on accuracy metrics (R² score).
  
### 🧑‍💻 Technical Indicators
- Includes popular **financial indicators**:
  - **SMA (Simple Moving Average)**
  - **EMA (Exponential Moving Average)**
  - **RSI (Relative Strength Index)**
  - **Bollinger Bands (BB)**
- Toggle the display of these indicators with adjustable parameters.

### 💼 Portfolio Simulation
- Simulate your **portfolio performance** based on stock price forecasts.
- Evaluate performance using **Sharpe Ratio**, **Monte Carlo simulations**, and forecasted portfolio values.

### 📉 Confidence Interval Visualization
- **Confidence intervals** (upper and lower bounds) displayed alongside predictions from the **Prophet** model.
- Option to toggle confidence intervals on or off from the sidebar.

---

## 🛠 Technologies Used

- **Streamlit**: Web app framework for interactive dashboards.
- **Prophet**: Forecasting tool by Facebook for time-series data with confidence intervals.
- **AutoReg**: Auto-regressive time series forecasting.
- **Plotly**: Visualization library for interactive graphs and plots.
- **Scikit-learn**: Machine learning library for metrics and model evaluations.
- **Pandas**: Data manipulation and cleaning.
- **NumPy**: Numerical operations on data.

---

## 🔧 Installation

### Requirements

- **Python 3.x**
- **Pip** (Python package installer)

### Steps to Run Locally

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/stock-forecasting-app.git
   cd stock-forecasting-app
   ```

2. **Install the Required Packages**:
   Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
   ```

   Then install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**:
   Start the app locally with:
   ```bash
   streamlit run app.py
   ```

4. **Access the App**:
   After running the app, it will be available in your browser at: `http://localhost:8501`.

---

## 📝 Features in Detail

### Stock Data Loading & Preprocessing
The app loads stock data from a **Parquet** file, cleans it, and prepares it for forecasting. Missing values are interpolated, and the data is resampled daily.

### Forecast Models
Two forecasting models are implemented:
- **AutoReg (Auto-Regressive Model)**: Predicts future stock prices based on past data points.
- **Prophet**: Uses seasonal trends and holidays to forecast future stock prices, including confidence intervals for prediction uncertainty.

### Portfolio Simulator
Simulate the performance of your portfolio with stock forecasts and:
- Set your **investment amount**.
- Track the **portfolio value** over time.
- Perform **Monte Carlo simulations** to assess potential risk.
  
### Interactive Forecast & Historical Plots
The app allows you to:
- Compare **forecasted values** with actual stock prices using interactive line charts.
- Display the **confidence intervals** from Prophet forecasts.
  
---

## 🚀 Future Enhancements
- **Advanced ML Models**: Add more complex models like **XGBoost** and **LSTM** for improved accuracy.
- **User Authentication**: Secure user login for personalized portfolio management.
- **Real-time Data Integration**: Integrate **live stock price feeds** from financial APIs.
- **Advanced Financial Metrics**: Add features such as **Max Drawdown** and **Sortino Ratio**.

---

## 🤝 Contributing

We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new feature branch: `git checkout -b feature-branch`.
3. Commit changes: `git commit -m 'Add new feature'`.
4. Push to your branch: `git push origin feature-branch`.
5. Create a pull request.

---

## 🛡 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

For any questions, feedback, or collaboration inquiries, please reach out:

**Name**: Lovish Dhanda 
**Email**: dhandal@student.douglascollege.ca
