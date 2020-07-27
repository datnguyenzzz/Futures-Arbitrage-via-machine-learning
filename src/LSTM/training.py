import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

binance_exchanges = ['BTCUSDT','ETHUSDT']
bitmex_exchanges = ['XBTUSD','ETHUSD']
hitbtc_exchanges = ['BTCUSD','ETHUSD']
markets = ['Binance','Bitmex','Hitbtc']

data_path = 'D:\My_Code\database\Futures_summer_2020\preprocessed_data.csv'

data = pd.read_csv(data_path)
data = data.drop(columns=['Unnamed: 0'])

"""
input layer (amount_of_stocks, n - lag_order, lag_order, exogenous_features)
dense layer (n - lag_order, amount_of_stocks)
"""
def lstm_model(lag_order: int,
               learning_rate: int,
               individual_output_dim: int,
               epochs: int = 200,
               batch_size: int = 100,
               combined_output_dim: int = 6, #= amount of stock
               dropout_rate: float = 0.1,
               exogenous_features: int = 4,
               percentile: int = 10):

    #split data

    n,d = data.shape
    amount_of_stocks = 6

    #input layer
    X = np.zeros((n - lag_order, amount_of_stocks, lag_order, exogenous_features))
    #dense layer
    Y = np.zeros((n - lag_order, amount_of_stocks))

    #return spread open volume

    for i in range(amount_of_stocks):
        for j in range(n - lag_order):
            for k in range(exogenous_features):
                ind = i * exogenous_features + k
                X[j,i,:,k] = data.values[j : (j+lag_order), ind]

            Y[j,i] = data.values[j+lag_order, i * exogenous_features]

    #print('X= ',X)
    #print('Y= ',Y)

    X_train = X[0 : int((n - lag_order) * 0.7)]
    Y_train = Y[0 : int((n - lag_order) * 0.7)]

    X_val = X[int((n - lag_order) * 0.7) : int((n - lag_order) * 0.85)]
    Y_val = Y[int((n - lag_order) * 0.7) : int((n - lag_order) * 0.85)]

    X_test = X[int((n - lag_order) * 0.85) : ]
    Y_test = Y[int((n - lag_order) * 0.85) : ]


lstm_model(lag_order=2, learning_rate= 0.001, individual_output_dim=1)
