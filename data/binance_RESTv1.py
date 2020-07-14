import urllib.parse
import requests
import json

class RestApiReq():
    def __init__(self):
        self.method = ""
        self.url = ""
        self.host = ""
        self.post_body = ""
        self.header = dict()
        self.header.update({"client_SDK_Version": "binance_dutures-1.0.1-py3.7"})

class binanceRESTApi():
    def __init__(self,baseUrl,apiKey,secretKey):
        self.baseUrl = baseUrl
        self.apiKey = apiKey
        self.secretKey = secretKey

    def call_sync(self,request):
        if request.method == "GET":
            print(request.host + request.url)
            response = requests.get(request.host + request.url, headers=request.header)
            return response.text

    def create_get_request(self,url,encoded):
        request = RestApiReq()
        request.method = "GET"
        request.host = self.baseUrl
        request.header.update({'Content-Type': 'application/json'})
        request.url = url + "?" + encoded
        return request

    def get_candlestick_data(self,symbol,interval,startTime,endTime,limit):
        param = dict()

        def put(name,value):
            if value is not None:
                if isinstance(value, list):
                    param[name] = json.dumps(value)
                elif isinstance(value, float):
                    param[name] = ('%.20f' % (value))[slice(0, 16)].rstrip('0').rstrip('.')
                else:
                    param[name] = str(value)

        put("symbol",symbol)
        put("interval",interval)
        put("startTime",startTime)
        put("endTime",endTime)
        put("limit",limit)

        if len(param) == 0:
            encoded = ""
        else:
            encoded = urllib.parse.urlencode(param)

        request = self.create_get_request("/fapi/v1/klines",encoded)

        response = self.call_sync(request)

        return response
