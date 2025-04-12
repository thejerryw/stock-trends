# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf

start = '2010-01-01' # start date for data
end = '2024-12-31' #end date for data

df = yf.download('AAPL', start=start, end=end) # download data using yfinance
print(df.head()) # printing data for start dates
print(df.tail()) # printing data for end dates

df = df.reset_index() # changes from date -> 0 1 2 3 4 etc.
print(df.head())

df = df.drop(['Date'], axis = 1) # removes date and adj close from the table
print(df.head())

plt.plot(df.Close) # plots the closing data 
print(df)

ma100 = df.Close.rolling(100).mean() # takes moving average of last 100 closing data 
print(ma100)

plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r') # plots moving average of last 100 days [smooth]

ma200 = df.Close.rolling(200).mean() # takes moving average of last 200 closing data 
print(ma200)

plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r') 
plt.plot(ma200, 'r') # plots moving average of last 200 days [smooth]

print(df.shape)

#splitting data into training and testing

data_training = pd.DataFrame(df['Close'][0 : int(len(df)*0.70)]) # splits first part of closing data (0-70%)
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))]) # second part of closing data (70-100%)

print(data_training.shape)
print(data_testing.shape)

print(data_training.head())
print(data_testing.head())

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1)) # scales all data to range (0 to 1)

data_training_array = scaler.fit_transform(data_training) # finds min and max of data, computing and scaling all values using it
print(data_training_array)

x_train = []
y_train = []

for i in range (100, data_training_array.shape[0]): # sliding window of 100 steps
    x_train.append(data_training_array[i-100 : i]) # stores the last 100 data points for each step
    y_train.append(data_training_array[i, 0]) # stores value after 100 data points [label]

x_train, y_train = np.array(x_train), np.array(y_train) # convert to numpy arrays [which keras/tensorflow expect as input]

# ML model

from keras.layers import Input, Dense, Dropout, LSTM
from keras.models import Sequential 

model = Sequential() # initializes new model for layers to be added
model.add(Input(shape=(x_train.shape[1], 1))) # input layer (time steps, num. of features per ts)
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True)) # 50 neurons, relu activation, passes output onto next LSTM layer
model.add(Dropout(0.2)) # drops 20% of units randomly to avoid overfitting

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True)) 
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 50, activation = 'relu', return_sequences = False))
model.add(Dropout(0.4))

model.add(Dense(units = 1)) # output layer, predicting single value [closing price]
print(model.summary()) 

model.compile(optimizer = 'adam', loss = 'mean_squared_error') # adam is adaptive optimizer, MSE loss for regression
model.fit(x_train, y_train, epochs = 50) # trains model on x and y train for 50 epochs [passes]
model.save('keras_model.keras')

# now doing same steps with test data

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range (100, input_data.shape[0]):
    x_test.append(input_data[i-100 : i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# making predictions

y_predicted = model.predict(x_test) # predicting y using x_test data

y_predicted = scaler.inverse_transform(y_predicted) # scaling back predicted data using original scale
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)) # scaling back test data using original scale

# plotting predicted vs. actual data

plt.figure(figsize = (12, 6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()