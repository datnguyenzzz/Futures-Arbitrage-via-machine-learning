import binance_RESTv1 as binance
from datetime import datetime

baseUrl="https://fapi.binance.com"
apiKey=""
secretKey=""

client = binance.binanceRESTApi(baseUrl,apiKey,secretKey)

strStart = "2020-01-01 00:00:00"
startTime = datetime.timestamp(datetime.strptime(strStart, '%Y-%m-%d %H:%M:%S'))
strEnd = "2020-07-14 00:00:00"
endTime = datetime.timestamp(datetime.strptime(strEnd, '%Y-%m-%d %H:%M:%S'))

res = client.get_candlestick_data(symbol="BTCUSDT", interval="1m",
                                  startTime=None, endTime=None, limit=100)


print(res)
