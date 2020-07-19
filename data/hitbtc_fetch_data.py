from hitbtcapi.client import Client
from datetime import datetime
import pandas as pd

api_key = 'EdUbs4V_gDOIZp295e8tmWf_gniIWAnB'
api_secret = '1kc2sVOoXVGSjsPK5kat80O2pF5HhIKX'

client = Client(api_key,api_secret)

finalDF = pd.DataFrame()

Tstart = datetime.timestamp(datetime.strptime("2019-01-01 00:00:00",'%Y-%m-%d %H:%M:%S'))
Tend = datetime.timestamp(datetime.strptime("2020-07-15 00:00:00",'%Y-%m-%d %H:%M:%S'))
#print(Tstart,Tend,Tend - Tstart)

while (Tstart < Tend):
    print(Tstart/Tend)
    TstartNext = Tstart + 3600

    params = {
        'from': datetime.fromtimestamp(Tstart),
        'till': datetime.fromtimestamp(TstartNext)
    }

    data = pd.DataFrame(client.get_candles('ETHUSD', limit=1000, period='M1', **params))

    finalDF = pd.concat([finalDF,data])

    Tstart = TstartNext + 60


print(finalDF)
finalDF.to_csv('Hitbtc-ETHUSD-1m-data.csv')
