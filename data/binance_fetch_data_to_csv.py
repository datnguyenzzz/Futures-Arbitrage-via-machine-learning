import binance_RESTv1 as binance
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt

baseUrl="https://fapi.binance.com"
apiKey=""
secretKey=""

client = binance.binanceRESTApi(baseUrl,apiKey,secretKey)

strStart = 1534107600

finalDF = pd.DataFrame()

now = datetime.timestamp(datetime.now())

time = 0
while (1==1):
    startTime = 1568062800 + 86400*time
    endTime = 1568062800 + 86400*(time+1)

    if (endTime > now):
        print('end game')
        break

    res = client.get_candlestick_data(symbol="BTCUSDT", interval="1m",
                                      startTime=startTime*1000, endTime=endTime*1000, limit=1500)

    df = pd.DataFrame(res)
    df = df.drop(columns=[0,6,7,8,9,10,11])
    for i in range(1,6):
        df[i] = df[i].astype(float)
    df = df.set_axis(['Open','High','Low','Close','Volume'],axis=1, inplace=False)
    df = df.rename(index = lambda s: s+1440*time)

    finalDF = pd.concat([finalDF,df])

    time+=1

finalDF.to_csv('binance_data.csv',index=False,header=False)

finalDF['Close'].plot(kind='line')
plt.show()
print(finalDF)
