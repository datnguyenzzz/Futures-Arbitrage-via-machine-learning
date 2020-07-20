import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR

data_path = 'D:\My_Code\database\Futures_summer_2020\preprocessed_data.csv'

data = pd.read_csv(data_path)
data = data.drop(columns=['Unnamed: 0'])

#data['Hitbtc_ETHUSD_open'].plot(kind='line')
#plt.show()

n = np.shape(data)[0]
train_data = data[0 : int(0.7 * n)]


"""
#model

model = VAR(train_data)
results = model.fit(1)

params = results.params.values

print(params.shape)


df = pd.DataFrame(params)
df.to_csv("D:\My_Code\database\Futures_summer_2020\output\VAR\model_prediction.csv")
"""


params = pd.read_csv("D:\My_Code\database\Futures_summer_2020\output\VAR\model_prediction.csv")
params = params.drop(columns=['Unnamed: 0'])

print(params)

validation_data = data[int(0.7 * n) : int(0.85 * n)]
test_data = data[int(0.85 * n) : ]

val_start = validation_data.index.min()
val_end = validation_data.index.max()

prediction = np.zeros((len(validation_data), 24)).astype(float)

#split to A
A0 = params.iloc[0,:].to_numpy().reshape(24,1)
n_features = 24

p_lags = 1

#print(params.iloc[(1-1) * n_features + 1 : 1 * n_features + 1, : ])



for i in range(val_start,val_end+1):

    Yi = params.iloc[0,:].to_numpy().reshape(n_features,1).astype(float)

    for p in range(1, p_lags+1):
        j = i - p
        if (j < val_start):
            Yj = data.iloc[j, :].to_numpy().reshape(n_features,1)
        else:
            Yj = prediction[j - val_start, :].reshape(n_features,1)
        Yi += np.dot(params.iloc[(p-1) * n_features + 1 : p * n_features + 1, : ].to_numpy().reshape(n_features,n_features), Yj)

    prediction[i-val_start, :] = Yi.reshape(1,n_features)



print(prediction)
