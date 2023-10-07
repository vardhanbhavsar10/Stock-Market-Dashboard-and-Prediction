import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
st.title('Stock Dashboard')
defaultCompany = "TSLA"
companyTicker = st.text_input('Stock Ticker:', defaultCompany)
defaultDate = date(2022, 10, 7)
startDate = st.date_input('Start Date:', defaultDate)
endDate = st.date_input('End Date:')
data = yf.download(companyTicker, start=startDate, end=endDate)

try:
    #select box to choose an indicator
    indicator = st.selectbox("Select Indicator",
                             ["None", "SMA (Simple Moving Average)", "EMA (Exponential Moving Average)"])

    if indicator != "None":
        window = st.slider("Select Window Size for Indicator", min_value=2, max_value=len(data) // 2, value=10)

        if indicator == "SMA (Simple Moving Average)":
            data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()
        elif indicator == "EMA (Exponential Moving Average)":
            data[f'EMA_{window}'] = data['Close'].ewm(span=window, adjust=False).mean()

    line, candle, area,forecast = st.tabs(['Line Chart', 'Candle Chart', 'Area Chart','Forecast Data'])

    # Three different tabs for Open, Close, and Adj Close
    with line:
        # In line graph three different parameters of open close and adj close
        open, close, adjClose = st.tabs(['Open', 'Close', 'Adj Close'])
        with open:
            chart = go.Figure()
            if indicator != "None":
                chart.add_trace(go.Scatter(x=data.index, y=data[f'{indicator.split()[0].upper()}_{window}'],
                                           name=f'{indicator} ({window} Days)'))
            chart.add_trace(go.Scatter(x=data.index, y=data['Open'], name='Open Price'))
            chart.update_layout(title_text='Line Plot', xaxis_rangeslider_visible=False)
            st.plotly_chart(chart)
        with close:
            chart = go.Figure()
            if indicator != "None":
                chart.add_trace(go.Scatter(x=data.index, y=data[f'{indicator.split()[0].upper()}_{window}'],
                                           name=f'{indicator} ({window} Days)'))
            chart.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'))
            chart.update_layout(title_text='Line Plot', xaxis_rangeslider_visible=False)
            st.plotly_chart(chart)
        with adjClose:
            chart = go.Figure()
            if indicator != "None":
                chart.add_trace(go.Scatter(x=data.index, y=data[f'{indicator.split()[0].upper()}_{window}'],
                                           name=f'{indicator} ({window} Days)'))
            chart.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'))
            chart.update_layout(title_text='Line Plot', xaxis_rangeslider_visible=False)
            st.plotly_chart(chart)
    with candle:
        fig = go.Figure()
        # length of candle stick graph represent how much value increased or decreased based on prev close
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],increasing_line_color= 'green', decreasing_line_color= 'red'))
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig)
    with area:
        areaChart = st.area_chart(data,y='Close')
    with forecast:
        n_years = st.slider('Years of prediction:', 1, 4)
        period = n_years * 365
        date = data.index
        df_train = data[['Close']]
        df_train['Date']=date
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period, freq='D')
        forecast = m.predict(future)
        # Show and plot forecast
        st.subheader('Forecast data')
        st.write(f'Forecast plot for {n_years} years')
        fig = plot_plotly(m, forecast)
        fig.update_layout(title='Prophet Forecast with Actual Data',
                          xaxis_title='Date',
                          yaxis_title='Value')
        st.plotly_chart(fig)
    st.header(f"Price Data from {startDate} to {endDate}:")
    st.write(data)
except ValueError:
    st.error("Enter Valid Date Range or Company")

