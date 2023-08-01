#!/usr/bin/env python
# coding: utf-8

# In[3]:


import yfinance as yf
stock=input("Enter the stock name: ")
st_data=yf.Ticker(stock)
st_data=st_data.history(period="max")


# In[4]:


st_data


# In[11]:


st_data.plot.line(y="Close", use_index=True)


# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[13]:


st_data.head()


# In[15]:


stockinfo=st_data["Close"]
training_set=pd.DataFrame(stockinfo)


# In[16]:


training_set.head()


# In[17]:


sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)


# In[18]:


X_train = []
y_train = []
for i in range(60, 2035):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[19]:


pip install keras


# In[20]:


get_ipython().system('pip install tensorflow')


# In[21]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense


# In[22]:


model = Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(X_train,y_train,epochs=100,batch_size=32)


# In[24]:


predicted_stock_price = model.predict(X_train)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# In[25]:


plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[ ]:




