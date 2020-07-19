import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = 'D:\My_Code\database\Futures_summer_2020\preprocessed_data.csv'

data = pd.read_csv(data_path)
data['Hitbtc_ETHUSD_open'].plot(kind='line')
plt.show()
