import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import zero_one_loss

prediction_path = 'D:/My_Code/database/Futures_summer_2020/output/VAR/prediction_var.csv'

binance_exchanges = ['BTCUSDT','ETHUSDT']
bitmex_exchanges = ['XBTUSD','ETHUSD']
hitbtc_exchanges = ['BTCUSD','ETHUSD']
markets = ['Binance','Bitmex','Hitbtc']

prediction_data = pd.read_csv(prediction_path)

def calc(prediction_data):

    PNL_direct = np.zeros(len(header)).astype('float32')
    PNL_inverse = np.zeros(len(header)).astype('float32')

    for type in range(len(header)):
        print("Exchange :",header[type])

        """We had 100 contracts for each exchange"""

        pred_return = prediction_data[header[type]+'_return'].to_numpy()

        data_path = "D:/My_Code/database/Futures_summer_2020/" + header[type] + "-1m-data.csv"

        data = pd.read_csv(data_path)
        test_data = data[int(data.shape[0] * 0.85) : ]

        real_open = test_data['open'].to_numpy()
        real_close = test_data['close'].to_numpy()

        for i in range(pred_return.shape[0]):
            if (pred_return[i] > 0):
                diff = 1
            else:
                diff = -1
            PNL_inverse[type] += diff * (1/real_open[i] - 1/real_close[i]) * real_close[i] * 100
            PNL_direct[type] += -diff * (real_open[i] - real_close[i]) * 100

        print("Profit with direct future: ", PNL_direct[type])
        print("Profit with inverse future: ", PNL_inverse[type])
    print("**************************************")
    print("Direct future sharpe ratio: ", np.mean(PNL_direct) / np.std(PNL_direct))
    print("Inverse future sharpe ratio: ", np.mean(PNL_inverse) / np.std(PNL_inverse))
    print("**************************************")


header = []
for market in markets:
    if (market=='Binance'):
        tmp = binance_exchanges
    elif (market=='Bitmex'):
        tmp = bitmex_exchanges
    else:
        tmp = hitbtc_exchanges

    for exchange in tmp:
        header.append(market+'_'+exchange)

print("VAR model")
print("**************************************")
calc(prediction_data=prediction_data)

print("LSTM model")
print("**************************************")
prediction_path = 'D:/My_Code/database/Futures_summer_2020/output/LSTM/prediction_LSTM_05.csv'
prediction_data = pd.read_csv(prediction_path)
calc(prediction_data=prediction_data)
