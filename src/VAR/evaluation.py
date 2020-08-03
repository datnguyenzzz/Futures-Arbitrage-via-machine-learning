import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import zero_one_loss

data_path = 'D:\My_Code\database\Futures_summer_2020\preprocessed_data.csv'
prediction_path = 'D:/My_Code/database/Futures_summer_2020/output/VAR/test_prediction_var1.csv'

binance_exchanges = ['BTCUSDT','ETHUSDT']
bitmex_exchanges = ['XBTUSD','ETHUSD']
hitbtc_exchanges = ['BTCUSD','ETHUSD']
markets = ['Binance','Bitmex','Hitbtc']

real_data = pd.read_csv(data_path)
prediction_data = pd.read_csv(prediction_path)

#optiomal model by validtaion data

n = np.shape(real_data)[0]
validation_data = real_data[int(0.7 * n) : int(0.85 * n)]

val_start = validation_data.index.min()
val_end = validation_data.index.max()

#prediction_data = prediction_data.rename(index = lambda s: s+val_start)

path_return = 'Binance_BTCUSDT_return'
path_open = 'Binance_BTCUSDT_open'
path_close = 'Binance_BTCUSDT_close'

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

MSE_values = np.zeros(len(header_return))

for type in range(len(header_return)):


    """MSE Evaluate"""
    path_return = header_return[type]

    real_return = validation_data[path_return].to_numpy()
    pred_return = prediction_data[path_return].to_numpy()

    for i in range(len(real_return)):
        MSE_values[type] += 1/len(real_return) * (real_return[i] - pred_return[i])**2

    """
    """"""

    pred_close = np.zeros(validation_data.shape[0])
    real_close = np.zeros(validation_data.shape[0])
    real_open = validation_data[path_open].to_numpy()

    for i in range(1,len(real_return)):
        pred_close[i-1] = (pred_return[i] + 1) * real_open[i]
    for i in range(len(real_return)):
        real_close[i] = (real_return[i] + 1) * real_open[i]

    print('real_close: ',real_close)
    print('pred_close: ',pred_close)

    print('real_return: ', real_return)
    print('pred_return: ', pred_return)

    print('MSE: ',MSE)

    ax = plt.gca()

    realDF = pd.DataFrame(real_close,columns=['real'])
    predDF = pd.DataFrame(pred_close,columns=['prediction'])
    realDF.plot(kind='line',color="y",ax=ax, y='real')
    predDF.plot(kind='line',color="r",ax=ax, y='prediction')

    plt.savefig('VAR_closing_price.png')
    plt.show()
    """

print("Average MSE among all exchanges: ", np.mean(MSE_values))

real_return = validation_data[header_return].to_numpy()
pred_return = prediction_data[header_return].to_numpy()

real_ACC = real_return.reshape(real_return.shape[0] * real_return.shape[1])
real_ACC[real_ACC > 0] = 1
real_ACC[real_ACC < 0] = 0


pred_ACC = pred_return.reshape(pred_return.shape[0] * pred_return.shape[1])
pred_ACC[pred_ACC > 0] = 1
pred_ACC[pred_ACC < 0] = 0


print("VAR Accuracy: ", 1 - zero_one_loss(real_ACC,pred_ACC))
