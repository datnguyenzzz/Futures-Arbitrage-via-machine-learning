import kraken_WSv1 as krakenWsApi

apiPath = "wss://futures.kraken.com/ws/v1"
apiKey = "g84thgQEkIYflA3cCFuI7Jf8zFytMasoSyyAW87IpoPtQydFsmsxu/Hg"
privateKey = "gp/0DCpaR6h1segtkW/gQuRSrI8SZq2fqVoO8I+mYmuBMtnxqjzC9NeyolvZbsR1iDkR3c5mtELCVDxpdz3skQ=="
timeout = 10
trace = False

ws = krakenWsApi.KrakenWSMethods(apiPath,apiKey,privateKey,timeout,trace)

def subscribe():
    productIds = ["PI_XBTUSD"]

    feed = "ticker"
    ws.subscribe_public(feed,productIds)

def unsubscribe():
    productIds = ["FV_XRPXBT_180615"]

    feed = "ticker"
    ws.unsubscribe_public(feed,productIds)

input()
subscribe()
input()
unsubscribe()
exit()
