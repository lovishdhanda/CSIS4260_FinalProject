# ================== ENHANCED STOCK FORECASTING STREAMLIT APP ==================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========================== Theme Toggle ==========================
theme_mode = st.sidebar.radio("\U0001F319 Theme Mode", ["Light Mode", "Dark Mode"])
theme = "plotly_dark" if theme_mode == "Dark Mode" else "plotly_white"

# ========================== Load Data ==========================
@st.cache_data
def load_stock_data(file_path):
    df = pd.read_parquet(file_path)
    df.columns = df.columns.str.strip().str.capitalize()
    df.rename(columns={'Company': 'name'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.set_index('Date', inplace=True)
    df = df.sort_index()

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numeric_cols + ['name'], inplace=True)
    return df

df = load_stock_data("scaled_dataset_1x_snappy.parquet")

# ========================== Valid Company Filter ==========================
min_days = 126
valid_companies = [c for c in df['name'].unique() if len(df[df['name'] == c]) >= min_days]
df = df[df['name'].isin(valid_companies)]

# ========================== Sidebar UI ==========================
st.sidebar.header("\U0001F4CA Forecast Settings")
selected_companies = st.sidebar.multiselect("Select Companies", valid_companies, default=valid_companies[:1])
forecast_days = st.sidebar.slider("Forecast Days", 10, 126, step=5)
investment_amount = st.sidebar.number_input("Investment Amount", min_value=100.0, value=1000.0)
st.sidebar.markdown("---")

# Technical Indicator Toggles
st.sidebar.header("Technical Indicator Settings")
show_sma = st.sidebar.checkbox("Show SMA", True)
show_ema = st.sidebar.checkbox("Show EMA", True)
show_rsi = st.sidebar.checkbox("Show RSI", True)
show_bb = st.sidebar.checkbox("Show Bollinger Bands", True)
sma_window = st.sidebar.slider("Window for SMA/EMA", 10, 50, value=20)

# Confidence Interval Toggle
show_confidence_interval = st.sidebar.checkbox("Show Confidence Interval", True)

# ========================== UI Tabs ==========================
tab1, tab2, tab3, tab4 = st.tabs(["\U0001F4C8 Price Trend", "\U0001F52E Forecast", "\U0001F4B0 Portfolio", "\U0001F4CA Indicators"])

portfolio_df_list = []

for company in selected_companies:
    company_data = df[df['name'] == company].copy()
    company_data = company_data[~company_data.index.duplicated()]
    numeric_data = company_data.select_dtypes(include='number')
    company_data_cleaned = numeric_data.resample('D').mean().interpolate()
    company_data_cleaned['name'] = company

    y = company_data_cleaned['Close'].dropna().values

    # ========================== AutoReg ==========================
    best_ar_lag, best_r2, best_preds = 1, -np.inf, None
    for lag in range(1, 31):
        try:
            train = y[-(len(y) - lag):]
            model = AutoReg(train, lags=lag, old_names=False).fit()
            preds = model.forecast(steps=forecast_days)
            r2 = r2_score(y[-forecast_days:], preds[:len(y[-forecast_days:])])
            if r2 > best_r2:
                best_r2, best_ar_lag, best_preds = r2, lag, preds
        except Exception:
            continue

    autoreg_model = AutoReg(y[-(len(y) - best_ar_lag):], lags=best_ar_lag, old_names=False).fit()
    autoreg_preds = autoreg_model.forecast(steps=forecast_days)

    autoreg_df = pd.DataFrame({
        'Date': pd.date_range(start=company_data_cleaned.index[-1] + pd.Timedelta(days=1), periods=forecast_days),
        'Predicted_Close': np.round(autoreg_preds, 2)  # Round to two decimal points
    })

    # Prophet Forecast
    prophet_df = company_data_cleaned.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    prophet_model = Prophet(daily_seasonality=True, interval_width=0.95)
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(periods=forecast_days)
    forecast = prophet_model.predict(future)
    forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).rename(columns={'ds': 'Date', 'yhat': 'Predicted_Close', 'yhat_lower': 'Lower_CI', 'yhat_upper': 'Upper_CI'})

    # Evaluation
    true_vals = y[-forecast_days:] if len(y) >= forecast_days else y
    ar_eval = r2_score(true_vals, autoreg_preds[:len(true_vals)])
    pr_eval = r2_score(true_vals, forecast_result['Predicted_Close'].values[:len(true_vals)])

    best_model = "AutoReg" if ar_eval >= pr_eval else "Prophet"
    forecast_df = autoreg_df if best_model == "AutoReg" else forecast_result
    forecast_df['Company'] = company

    # Portfolio Simulation
    current_price = company_data_cleaned['Close'].iloc[-1]
    shares = investment_amount / current_price / len(selected_companies)
    forecast_df['Portfolio Value'] = forecast_df['Predicted_Close'] * shares
    portfolio_df_list.append(forecast_df)

    # ========== Tab 1: Historical Charts ==========
    with tab1:
        st.subheader(f"{company} - Price Chart")
        fig = px.line(company_data_cleaned, x=company_data_cleaned.index, y="Close", title=f"{company} Close Price", template=theme)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"{company} - Candlestick Chart")
        fig_candle = go.Figure(data=[go.Candlestick(
            x=company_data_cleaned.index,
            open=company_data_cleaned['Open'], 
            high=company_data_cleaned['High'],
            low=company_data_cleaned['Low'], 
            close=company_data_cleaned['Close'], 
            name="Candlestick"
        )])
        fig_candle.update_layout(
            template=theme,
            xaxis_rangeslider_visible=True,  # Ensures the chart is interactive
            title=f"{company} Candlestick Chart"
        )
        st.plotly_chart(fig_candle, use_container_width=True)

    # ========== Tab 2: Forecast + Overlap ==========
    with tab2:
        st.subheader(f"{company} Forecast ({forecast_days} Days) using {best_model}")
        st.dataframe(forecast_df[['Date', 'Predicted_Close']])

        fig_forecast = px.line(forecast_df, x='Date', y='Predicted_Close', title=f"Forecast - {company}", template=theme)
        st.plotly_chart(fig_forecast, use_container_width=True)

        # Overlap Plot
        overlap_df = company_data_cleaned[['Close']].reset_index().tail(forecast_days)
        overlap_df['Date'] = overlap_df['Date'].dt.date  # Convert to date object

        # Ensure forecast_df['Date'] is also in date format
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date']).dt.date

        # Now safely merge
        overlap_df = pd.merge(overlap_df, forecast_df[['Date', 'Predicted_Close']], on='Date', how='inner')

        st.subheader(f"{company} Forecast Overlap")
        fig_overlap = px.line(overlap_df, x='Date', y=['Close', 'Predicted_Close'], title=f"Overlap Forecast - {company}", template=theme)
        st.plotly_chart(fig_overlap, use_container_width=True)

        # Show Confidence Interval if toggled
        if show_confidence_interval:
            fig_forecast_with_ci = px.line(forecast_result, x='Date', y='Predicted_Close', title=f"Forecast with Confidence Interval ({company})", template=theme)
            fig_forecast_with_ci.add_traces([
                go.Scatter(x=forecast_result['Date'], y=forecast_result['Lower_CI'], fill='tonexty', mode='none', name="Lower Confidence Interval"),
                go.Scatter(x=forecast_result['Date'], y=forecast_result['Upper_CI'], fill='tonexty', mode='none', name="Upper Confidence Interval")
            ])
            st.plotly_chart(fig_forecast_with_ci, use_container_width=True)

    # ========== Tab 4: Indicators ==========
    with tab4:
        st.subheader(f"{company} - Technical Indicators")
        if show_sma:
            company_data_cleaned[f'SMA_{sma_window}'] = company_data_cleaned['Close'].rolling(window=sma_window).mean()
            fig_sma = px.line(company_data_cleaned, x=company_data_cleaned.index, y=['Close', f'SMA_{sma_window}'], title="SMA", template=theme)
            st.plotly_chart(fig_sma, use_container_width=True)

        if show_ema:
            company_data_cleaned[f'EMA_{sma_window}'] = company_data_cleaned['Close'].ewm(span=sma_window).mean()
            fig_ema = px.line(company_data_cleaned, x=company_data_cleaned.index, y=['Close', f'EMA_{sma_window}'], title="EMA", template=theme)
            st.plotly_chart(fig_ema, use_container_width=True)

        if show_rsi:
            delta = company_data_cleaned['Close'].diff()
            up = delta.clip(lower=0).rolling(window=14).mean()
            down = -1 * delta.clip(upper=0).rolling(window=14).mean()
            rs = up / down
            company_data_cleaned['RSI'] = 100 - (100 / (1 + rs))
            fig_rsi = px.line(company_data_cleaned, x=company_data_cleaned.index, y='RSI', title="RSI", template=theme)
            st.plotly_chart(fig_rsi, use_container_width=True)

        if show_bb:
            sma = company_data_cleaned['Close'].rolling(window=sma_window).mean()
            std = company_data_cleaned['Close'].rolling(window=sma_window).std()
            company_data_cleaned['BB_upper'] = sma + 2 * std
            company_data_cleaned['BB_lower'] = sma - 2 * std
            fig_bb = px.line(company_data_cleaned, x=company_data_cleaned.index,
                             y=['Close', 'BB_upper', 'BB_lower'], title="Bollinger Bands", template=theme)
            st.plotly_chart(fig_bb, use_container_width=True)

# ========== Tab 3: Portfolio Simulator ==========
with tab3:
    st.subheader("Simulated Portfolio Value")
    full_portfolio_df = pd.concat(portfolio_df_list)
    summary_df = full_portfolio_df.groupby('Date')['Portfolio Value'].sum().reset_index()
    st.dataframe(summary_df.tail())
    fig_summary = px.line(summary_df, x='Date', y='Portfolio Value', title="Portfolio Forecast (All Selected Stocks)", template=theme)
    st.plotly_chart(fig_summary, use_container_width=True)

    # Monte Carlo Sim
    st.subheader("Monte Carlo Simulation")
    simulations = 1000
    returns = summary_df['Portfolio Value'].pct_change().dropna()
    mean_return = returns.mean()
    std_return = returns.std()
    np.random.seed(42)
    simulations_result = np.cumprod(np.random.normal(mean_return, std_return, (forecast_days, simulations)) + 1, axis=0)
    final_values = simulations_result[-1] * investment_amount
    fig_mc = px.histogram(final_values, nbins=50, title="Final Portfolio Value Distribution (Monte Carlo)", template=theme)
    st.plotly_chart(fig_mc, use_container_width=True)

    # Sharpe Ratio
    sharpe_ratio = (mean_return * 252) / (std_return * np.sqrt(252))
    st.metric("Estimated Sharpe Ratio", f"{sharpe_ratio:.2f}")
