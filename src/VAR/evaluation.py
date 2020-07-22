import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_path = 'D:\My_Code\database\Futures_summer_2020\preprocessed_data.csv'
prediction_path = 'D:/My_Code/database/Futures_summer_2020/output/VAR/test_prediction_var1.csv'

real_data = pd.read_csv(data_path)
prediction_data = pd.read_csv(prediction_path)

#optiomal model by validtaion data

n = np.shape(real_data)[0]
validation_data = real_data[int(0.7 * n) : int(0.85 * n)]

val_start = validation_data.index.min()
val_end = validation_data.index.max()

prediction_data = prediction_data.rename(index = lambda s: s+val_start)

path_return = 'Binance_BTCUSDT_return'
path_open = 'Binance_BTCUSDT_open'


ax = plt.gca()

#open_pred.plot(color="y",ax=ax)
#valid_data_open.plot(color="c",ax=ax)
print(prediction_data[path_return])
#prediction_data[path_return].plot(color="y",ax=ax)
validation_data[path_return].plot(color="r",ax=ax)
plt.show()






"""
open_pred = np.zeros(validation_data.shape[0])
real_data_return = real_data[path_return].to_numpy().astype(float)
real_data_open = real_data[path_open].to_numpy().astype(float)

pred_data_return = prediction_data[path_return].to_numpy().astype(float)
valid_data_open = validation_data[path_open].to_numpy().astype(float)

open_pred[0] = (real_data_return[val_start-1] + 1) * real_data_open[val_start-1]

for i in range(1,validation_data.shape[0]):
    open_pred[i] = (pred_data_return[i-1] + 1) * open_pred[i-1]


print(open_pred)
print(valid_data_open)

MSE = 0.0

for i in range(validation_data.shape[0]):
    MSE += 1.0 / validation_data.shape[0] * (open_pred[i] - valid_data_open[i])**2

print('VAR MSE: ', MSE)

open_pred = pd.DataFrame(open_pred)
valid_data_open = pd.DataFrame(valid_data_open)
"""
