import kraken_WSv1 as krakenWsApi

apiPath = "wss://futures.kraken.com/ws/v1"
apiKey = "g84thgQEkIYflA3cCFuI7Jf8zFytMasoSyyAW87IpoPtQydFsmsxu/Hg"
privateKey = "gp/0DCpaR6h1segtkW/gQuRSrI8SZq2fqVoO8I+mYmuBMtnxqjzC9NeyolvZbsR1iDkR3c5mtELCVDxpdz3skQ=="
timeout = 10
trace = False

ws = krakenWsApi.KrakenWSMethods(apiPath,apiKey,privateKey,timeout,trace)
