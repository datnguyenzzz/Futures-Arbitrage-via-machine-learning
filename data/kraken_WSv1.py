import json
import sys
import websocket

class KrakenWSMethods():
    """Web Socket Connector"""

    def __init__(self,base_url,api_key="",private_key="",timeout=5,trace=False):
        websocket.enableTrace(trace)
        self.base_url = base_url
        self.api_key = api_key
        self.private_key = private_key
        self.timeout = timeout

        self.ws = None
        self.original_challenge = None
        self.signed_challenge = None
        self.challenge_ready = False

        self.__connect()

print("done")
