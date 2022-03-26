# pip install streamlit prophet yfinance plotly
import streamlit as st    #streamlit - framework
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go  #interactive graph

START = "2020-03-17"   #start date
TODAY = date.today().strftime("%Y-%m-%d") #string formatting

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'TSLA')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4) #slider tool for diff years
period = n_years * 365

#cache data, no need to re-download
@st.cache

#load stock data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True) #date in the first col
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... complete!')

st.subheader('Raw data')
st.write(data.tail())  #panda data frame

# Plot raw data
def plot_raw_data():
	fig = go.Figure()  #create figure
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))   #add tracing function
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Facebook Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()  #create FBProphet model
m.fit(df_train)  #fit the df_train data
future = m.make_future_dataframe(periods=period) #create future data frame
forecast = m.predict(future)


st.subheader('Forecast data')
st.write(forecast.tail())

# plot the forecast data
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast) 
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)