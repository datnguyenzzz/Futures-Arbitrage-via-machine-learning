import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, Concatenate, Dense
from tensorflow.keras import Model, Input
import tensorflow.keras.optimizers as optimizer
import tensorflow.keras.backend as K

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
               epochs: int = 140,
               batch_size: int = 120,
               combined_output_dim: int = 6, #= amount of stock
               dropout_rate: float = 0.1,
               exogenous_features: int = 4):

    #split data

    n,d = data.shape

    xxxxxxxdata = data[0 : int(0.01 * n)]
    xxxxxxxn = int(0.01 * n)

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

    x_train = X[0 : int((n - lag_order) * 0.7)]
    y_train = Y[0 : int((n - lag_order) * 0.7)]

    x_val = X[int((n - lag_order) * 0.7) : int((n - lag_order) * 0.85)]
    y_val = Y[int((n - lag_order) * 0.7) : int((n - lag_order) * 0.85)]

    x_test = X[int((n - lag_order) * 0.85) : ]
    y_test = Y[int((n - lag_order) * 0.85) : ]

    # Create network with 6 layer

    tensor = []
    nn_layer = []
    drop = []

    for i in range(amount_of_stocks):
        tensor.append(Input((lag_order, exogenous_features)))
        nn_layer.append(LSTM(individual_output_dim, return_sequences=True)(tensor[len(tensor)-1]))
        drop.append(Dropout(dropout_rate)(nn_layer[len(nn_layer)-1]))

    model_drop = Concatenate()(drop)
    full_LSTM = LSTM(combined_output_dim)(model_drop)
    full_drop = Dropout(dropout_rate)(full_LSTM)
    dense_layer = Dense(combined_output_dim,activation='softmax')(full_drop)

    LSTM_model = Model(inputs=tensor, outputs=dense_layer)

    RMSprop = optimizer.RMSprop(lr = learning_rate)

    """
    00 01
    def customized_loss(y_pred, y_true):
        num = K.sum(K.square(y_pred - y_true), axis=-1)
        y_true_sign = y_true > 0
        y_pred_sign = y_pred > 0
        logicals = K.equal(y_true_sign, y_pred_sign)
        logicals_0_1 = K.cast(logicals, 'float32')
        den = K.sum(logicals_0_1, axis=-1)
        return num/(1 + den)
    """
    """
    03
    def customized_loss(y_pred, y_true):
        residual =K.cast(y_true - y_pred,'float32')
        loss = tf.where(residual > 0, tf.multiply(K.square(residual),10.0), K.square(residual))
        return K.mean(loss)
    """
    """
    04-best
    def customized_loss(y_pred, y_true):
        penalty = 10.0
        loss = tf.where(tf.less(y_true - y_pred, 0), penalty * tf.square(y_true - y_pred) , tf.square(y_true - y_pred))
        num = K.sum(loss, axis=-1)

        y_true_sign = y_true > 0
        y_pred_sign = y_pred > 0
        logicals = K.equal(y_true_sign, y_pred_sign)
        logicals_0_1 = K.cast(logicals, 'float32')
        den = K.sum(logicals_0_1, axis=-1)
        return num/(0.5 + den)
    """
    """
    def customized_loss(y_pred, y_true):
        penalty = 10.0
        loss = tf.where(tf.less(y_true - y_pred, 0), penalty * tf.square(y_true - y_pred) , tf.square(y_true - y_pred))
        num = K.sum(loss, axis=-1)

        y_true_sign = y_true > 0
        y_pred_sign = y_pred > 0
        logicals = K.equal(y_true_sign, y_pred_sign)
        logicals_0_1 = K.cast(logicals, 'float32')
        den = K.sum(logicals_0_1, axis=-1)
        return num/(0.05 + den)
    """
    LSTM_model.compile(optimizer = RMSprop, loss='categorical_crossentropy')

    history = LSTM_model.fit([x_train[:,i,:,:] for i in range(amount_of_stocks)],
                              y_train, epochs=epochs, batch_size=batch_size,
                              validation_data=([x_val[:,i,:,:] for i in range(amount_of_stocks)] , y_val))

    prediction_returns = LSTM_model.predict([x_val[:,i,:,:] for i in range(amount_of_stocks)])

    print(history.history.keys())

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper right')
    plt.show()

    all_mse = []
    for i in range(amount_of_stocks):
        pred_return = prediction_returns[:,i].copy()
        real_return = y_val[:,i].copy()

        MSE = sum((pred_return - real_return)**2) / y_val.shape[0]
        all_mse.append(MSE)

    avg_mse = np.array(all_mse)
    return (np.mean(avg_mse) , prediction_returns)

################################################################################
"""
min_MSE = 1e9
random.seed()

lag_order_list = []
learning_rate_list = []
individual_output_dim_list = []
mean_MSE_list = []

for times in range(5):

    lag_order = random.randint(10,20)
    learning_rate = np.random.choice([0.01, 0.001, 0.0001, 0.00001, 0.05, 0.005, 0.0005, 0.00005])
    individual_output_dim = random.randint(1,40)

    print(lag_order,learning_rate,individual_output_dim)


    result,prediction_returns = lstm_model(lag_order=lag_order, learning_rate= learning_rate, individual_output_dim=individual_output_dim)

    lag_order_list.append(lag_order)
    learning_rate_list.append(learning_rate)
    individual_output_dim_list.append(individual_output_dim)
    mean_MSE_list.append(result)

    if (result < min_MSE):
        min_MSE = result
        result_lag_order = lag_order
        result_learning_rate = learning_rate
        result_individual_output_dim = individual_output_dim
        result_prediction_returns = prediction_returns

print('Minimum mean MSE is: ', min_MSE)
print('Minimum lag order is: ', result_lag_order)
print('Minimum learning rate is: ', result_learning_rate)
print('Minimum dim individual output is: ', result_individual_output_dim)
"""

result,prediction_returns = lstm_model(lag_order=25, learning_rate= 0.0001, individual_output_dim=1)

print('min MSE is: ', result)

header_return = []
for market in markets:
    if (market=='Binance'):
        tmp = binance_exchanges
    elif (market=='Bitmex'):
        tmp = bitmex_exchanges
    else:
        tmp = hitbtc_exchanges

    for exchange in tmp:
        header_return.append(market+'_'+exchange+'_return')

returnDF = pd.DataFrame(prediction_returns, columns = header_return)
returnDF.to_csv("D:/My_Code/database/Futures_summer_2020/output/LSTM/prediction_LSTM_05.csv")
"""
#all tries
df = pd.DataFrame( list(zip(individual_output_dim_list, lag_order_list, learning_rate_list, mean_MSE_list)),columns = ['Number of LSTM units', 'Lag order period', 'Learning rate', 'Mean MSE' ])
df.to_csv("D:/My_Code/database/Futures_summer_2020/output/LSTM/all_samples_LSTM1.csv")
"""
