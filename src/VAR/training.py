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


#model

model = VAR(train_data)
results = model.fit(1)

params = results.params.values

print(params.shape)
"""
validation_data = data[int(0.7 * n) : int(0.85 * n)]
test_data = data[int(0.85 * n) : ]

val_start = validation_data.index.min()
val_end = validation_data.index.max()
"""
print(params)
