import numpy as np
import pandas as pd

binance_exchanges = ['BTCUSDT','ETHUSDT']
bitmex_exchanges = ['XBTUSD','ETHUSD']
hitbtc_exchanges = ['BTCUSD','ETHUSD']
markets = ['Binance','Bitmex','Hitbtc']

def preprocess(data):
    """
    input data format: open high low close volume
    after preprocessing: close/open-1 , high-low, normalize open, normalize volume

    """
    print('preprocess is starting')

    n,m = data.shape
    new_data = np.zeros((n,m))

    n_features = len(binance_exchanges) + len(bitmex_exchanges) + len(hitbtc_exchanges)
    group_feature = int(m/n_features)

    for d in range(n_features):
        new_data[:,d * group_feature] = data.iloc[:,d * group_feature + 3] / data.iloc[:,d * group_feature] - 1

        new_data[:,d * group_feature + 1] = data.iloc[:,d * group_feature + 1] - data.iloc[:,d * group_feature + 2]
        spread_max,spread_min = np.max(new_data[:,d * group_feature + 1]), np.min(new_data[:,d * group_feature + 1])
        new_data[:,d * group_feature + 1] = (new_data[:,d * group_feature + 1] - spread_min) / (spread_max - spread_min)

        open_max,open_min = data.iloc[:,d * group_feature].max(), data.iloc[:,d * group_feature].min()
        new_data[:,d * group_feature + 2] = (data.iloc[:,d * group_feature] - open_min) / (open_max - open_min)

        volume_max,volume_min = data.iloc[:,d * group_feature + 4].max(), data.iloc[:,d * group_feature + 4].min()
        new_data[:,d * group_feature + 3] = (data.iloc[:,d * group_feature + 4] - volume_min) / (volume_max - volume_min)

    #print(new_data[1])
    header_data = []
    trash_data = []
    for market in markets:
        if (market=='Binance'):
            tmp = binance_exchanges
        elif (market=='Bitmex'):
            tmp = bitmex_exchanges
        else:
            tmp = hitbtc_exchanges

        for exchange in tmp:
            header_data.append(market+'_'+exchange+'_return')
            header_data.append(market+'_'+exchange+'_spread')
            header_data.append(market+'_'+exchange+'_open')
            header_data.append(market+'_'+exchange+'_volume')
            header_data.append(market+'_'+exchange+'_trash')
            trash_data.append(market+'_'+exchange+'_trash')

    finalDF = pd.DataFrame(new_data,columns=header_data)
    finalDF = finalDF.drop(columns=trash_data)
    return finalDF


def merge_data():
    finalDF = pd.DataFrame()

    nmin = 1000000
    for market in markets:
        if (market=='Binance'):
            tmp = binance_exchanges
        elif (market=='Bitmex'):
            tmp = bitmex_exchanges
        else:
            tmp = hitbtc_exchanges

        for exchange in tmp:
            path = market + '-' + exchange + '-1m-data.csv'

            dt = pd.read_csv(path)
            #print(path,dt['timestamp'][0])
            #print(path,dt['timestamp'][dt.shape[0]-1])

            print(dt.shape[0])
            dt = dt.drop(columns=['timestamp'])

            if market == 'Hitbtc':
                dt = dt.reindex(columns=['open','high','low','close','volume'])

            nmin = min(nmin,dt.shape[0])

            renamer = {'open': market+'_'+exchange+'_open',
                      'high': market+'_'+exchange+'_high',
                      'low': market+'_'+exchange+'_low',
                      'close': market+'_'+exchange+'_close',
                      'volume': market+'_'+exchange+'_volume',}

            dt = dt.rename(columns=renamer)
            finalDF = pd.concat([finalDF,dt],axis=1)


    return finalDF[0:nmin]


finalDF = merge_data()
p_finalDF =  preprocess(finalDF)
p_finalDF.to_csv('preprocessed_data.csv')
