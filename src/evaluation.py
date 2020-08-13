import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import zero_one_loss
import seaborn as sns

data_path = 'D:\My_Code\database\Futures_summer_2020\preprocessed_data.csv'
#prediction_path = 'D:/My_Code/database/Futures_summer_2020/output/VAR/prediction_var.csv'
prediction_path = 'D:/My_Code/database/Futures_summer_2020/output/LSTM/prediction_LSTM_03.csv'

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
    for i in range(min(len(real_return),len(pred_return))):
        MSE_values[type] += 1/min(len(real_return),len(pred_return)) * (real_return[i] - pred_return[i])**2


    """
    if (type==0):

        size = min(real_return.shape[0],pred_return.shape[0])

        pred_close = np.zeros(size)
        real_close = np.zeros(size)
        real_open = validation_data[path_open].to_numpy()

        for i in range(1,size):
            pred_close[i-1] = (pred_return[i] + 1) * real_open[i]
        for i in range(size):
            real_close[i] = (real_return[i] + 1) * real_open[i]

        ax = plt.gca()

        real_close = real_close[14312:14342]
        pred_close = pred_close[14312:14342]

        realDF = pd.DataFrame(real_close,columns=['real'])
        predDF = pd.DataFrame(pred_close,columns=['prediction'])
        realDF.plot(kind='line',color="y",ax=ax, y='real')
        predDF.plot(kind='line',color="r",ax=ax, y='prediction')

        plt.show()
    """
"""
print("Average MSE among all exchanges: ", np.mean(MSE_values))

real_return = validation_data[header_return].to_numpy()
pred_return = prediction_data[header_return].to_numpy()

size = min(real_return.shape[0] * real_return.shape[1], pred_return.shape[0] * pred_return.shape[1])

real_return = real_return[:size]
pred_return = pred_return[:size]

real_ACC = real_return.reshape(real_return.shape[0] * real_return.shape[1])
real_ACC[real_ACC > 0] = 1
real_ACC[real_ACC < 0] = 0


pred_ACC = pred_return.reshape(pred_return.shape[0] * pred_return.shape[1])
pred_ACC[pred_ACC > 0] = 1
pred_ACC[pred_ACC < 0] = 0
"""
for type in range(len(header_return)):
    path_return = header_return[type]

    real_return = validation_data[path_return].to_numpy()
    pred_return = prediction_data[path_return].to_numpy()

    size = min(real_return.shape[0], pred_return.shape[0])

    real_return = real_return[:size]
    pred_return = pred_return[:size]

    real_ACC = real_return.copy()
    real_ACC[real_ACC > 0] = 1
    real_ACC[real_ACC < 0] = 0

    TPR_list = [0.0]
    FPR_list = [0.0]
    acc = 0
    list_of_t = sorted(pred_return)

    for j in range(len(list_of_t)):
    #for j in range(5):
        t = list_of_t[j]

        pred_ACC = pred_return.copy()
        pred_ACC[pred_ACC > t] = 1
        pred_ACC[pred_ACC < t] = 0

        TP = np.sum(np.logical_and(pred_ACC == 1, real_ACC == 1))
        FP = np.sum(np.logical_and(pred_ACC == 1, real_ACC == 0))
        TN = np.sum(np.logical_and(pred_ACC == 0, real_ACC == 0))
        FN = np.sum(np.logical_and(pred_ACC == 0, real_ACC == 1))
        #print(TP,FP,TN,FN)
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        acc += (FPR-FPR_list[j]) * TPR
        TPR_list.append(TPR)
        FPR_list.append(FPR)

    sns.set(color_codes = True)
    plt.scatter(FPR_list,TPR_list, s=2, label=path_return + ', ACC: '+str(round(1 - acc, 3)))
    #plt.plot([0, 1], [0, 1], color='red')


plt.xlabel('False Positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curves')
plt.legend()
plt.show()



#print("Accuracy: ", 1 - zero_one_loss(real_ACC,pred_ACC))
