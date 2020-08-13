import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR

binance_exchanges = ['BTCUSDT','ETHUSDT']
bitmex_exchanges = ['XBTUSD','ETHUSD']
hitbtc_exchanges = ['BTCUSD','ETHUSD']
markets = ['Binance','Bitmex','Hitbtc']

data_path = 'D:\My_Code\database\Futures_summer_2020\preprocessed_data.csv'

data = pd.read_csv(data_path)
data = data.drop(columns=['Unnamed: 0'])

#data['Hitbtc_ETHUSD_open'].plot(kind='line')
#plt.show()

n = np.shape(data)[0]
train_data = data[0 : int(0.7 * n)]
validation_data = data[int(0.7 * n) : int(0.85 * n)]
test_data = data[int(0.85 * n) : ]

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

print(header_return)

train_return = train_data[header_return]
validtaion_return = validation_data[header_return]

#print(train_return)

#model validtaion

model = VAR(train_return)
results = model.fit(1)

lag_order = results.k_ar
train_return = train_return.to_numpy()
prediction = results.forecast(train_return[-lag_order:], validtaion_return.shape[0])

finalDF = pd.DataFrame(prediction,columns = header_return)

ax = plt.gca()

path_return = 'Binance_BTCUSDT_return'

print(finalDF)

finalDF.to_csv("D:/My_Code/database/Futures_summer_2020/output/VAR/prediction_var.csv")

#df.to_csv("D:\My_Code\database\Futures_summer_2020\output\VAR\model_prediction.csv")
