# Stock Price Prediction<br/>
This Python script aims to predict the stock price of a given company using historical stock price data. The prediction is performed using Long Short-Term Memory (LSTM) neural networks, a type of recurrent neural network (RNN) known for its ability to model sequential data.<br/>

# Requirements<br/>
To run the script, you need to have the following libraries installed:<br/>

yfinance<br/>
numpy<br/>
pandas<br/>
matplotlib<br/>
scikit-learn<br/>
keras<br/>
tensorflow<br/>

# Usage
Run the script and provide the name of the stock for which you want to predict the price. For example, you can input "AAPL" for Apple Inc. or "GOOGL" for Alphabet Inc.<br/>

The script will fetch historical stock price data using the Yahoo Finance API.<br/>

The data will be preprocessed, scaled, and split into training and testing sets.<br/>

An LSTM model will be trained on the training data for stock price prediction.<br/>

The script will then use the trained model to predict stock prices and plot the predicted stock prices against the actual stock prices.
