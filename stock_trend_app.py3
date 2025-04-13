import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import pandas_datareader as data 
import yfinance as yf
from keras.models import load_model
import streamlit as st

start = '2010-01-01' # start date for data
end = '2024-12-31' #end date for data

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter the Stock Ticker: ', 'AAPL') # getting user stock input for data after
df = yf.download(user_input, start=start, end=end) # download data using yfinance

# describing the data
st.subheader('Data from 2010 to 2024 on ' + user_input)
st.write(df.describe()) # summary of stock data in table

# visualizing the data
st.subheader(user_input + ' Closing Price Chart with 100-day Moving Average')
ma100 = df.Close.rolling(100).mean() # 100 day moving average
fig = plt.figure(figsize = (12, 6)) # plot figure
plt.plot(ma100, 'r') # plotting ma100 data
plt.plot(df.Close, 'b') # plotting closing price data
st.pyplot(fig)

st.subheader(user_input + ' Closing Price Chart with 100-day & 200-day Moving Averages')
ma100 = df.Close.rolling(100).mean() # 100 day moving average
ma200 = df.Close.rolling(200).mean() # 200 day moving average
fig = plt.figure(figsize = (12, 6)) # plot figure
plt.plot(ma100, 'r') # plotting ma100 data
plt.plot(ma200, 'g') # plotting ma200 data
plt.plot(df.Close, 'b') # plotting closing price data
st.pyplot(fig)

# splitting data into training and testing

data_training = pd.DataFrame(df['Close'][0 : int(len(df)*0.70)]) # splits first part of closing data (0-70%)
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))]) # second part of closing data (70-100%)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1)) # scales all data to range (0 to 1)

# loading the model

model = load_model('keras_model.h5')

# testing the data

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range (100, input_data.shape[0]):
    x_test.append(input_data[i-100 : i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# making predictions using testing data

y_predicted = model.predict(x_test) # predicting y using x_test data

y_predicted = scaler.inverse_transform(y_predicted) # scaling back predicted data using original scale
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)) # scaling back test data using original scale

# final observed vs. predicted plot

st.subheader('Predicted vs. Observed ' + user_input + ' Closing Price from 2010 to 2024')
fig2 = plt.figure(figsize = (12, 6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# making prediction on future closing price

quote = yf.download(user_input, start = '2010-01-01', end = '2025-04-11') # download data using yfinance
last_60_days = pd.DataFrame(quote['Close'][-60:])
last_60_scaled = scaler.fit_transform(last_60_days)
x_test_new = []
x_test_new.append(last_60_scaled)
x_test_new = np.array(x_test_new)
x_test_new = np.reshape(x_test_new, (x_test_new.shape[0], x_test_new.shape[1], 1)) # reshaping np array
pred_price = model.predict(x_test_new)
pred_price = scaler.inverse_transform(pred_price)
st.subheader(f'The Predicted Closing Price of {user_input} on 2025-04-14 is ${pred_price[0][0]:.2f}') # predicts next element in array
