import kraken_RESTv3 as krakenRestApi

apiPath = "https://futures.kraken.com/derivatives/api/v3"
apiKey = "g84thgQEkIYflA3cCFuI7Jf8zFytMasoSyyAW87IpoPtQydFsmsxu/Hg"
privateKey = "gp/0DCpaR6h1segtkW/gQuRSrI8SZq2fqVoO8I+mYmuBMtnxqjzC9NeyolvZbsR1iDkR3c5mtELCVDxpdz3skQ=="
timeout = 10


restPublic = krakenRestApi.KrakenApiMethods(apiPath,apiKey,privateKey,timeout)

def ApiCall():
    res = restPublic.get_tickers()
    print("tickers:\n", res)

ApiCall()
