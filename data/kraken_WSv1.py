import json
import hashlib
import base64
import hmac
import sys
import websocket

from time import sleep

class KrakenWSMethods():
    """Web Socket Connector"""

    def __init__(self,baseUrl,apiKey="",privateKey="",timeout=5,trace=False):
        websocket.enableTrace(trace)
        self.baseUrl = base_url
        self.apiKey = api_key
        self.privateKey = private_key
        self.timeout = timeout

        self.ws = None
        self.originalChallenge = None
        self.signedChallenge = None
        self.challengeReady = False

        self.__connect()

    def __subscribe_public(self,feed,productIds=None):
        """subscribe to give feed and product ids"""

        if productIds is None:
            requestMessage = {
                "event": "subscribe",
                "feed": feed
            }
        else:
            requestMessage = {
                "event": "subscribe",
                "feed": feed,
                "productIds": productIds
            }

        print("public subscribe to %s",feed)

        requestJson = json.dumps(requestMessage)
        self.ws.send(requestJson)

    def __unsubscribe_public(self,feed,productIds=None):
        """subscribe to give feed and product ids"""

        if productIds is None:
            requestMessage = {
                "event": "unsubscribe",
                "feed": feed
            }
        else:
            requestMessage = {
                "event": "unsubscribe",
                "feed": feed,
                "productIds": productIds
            }

        print("public unsubscribe to %s",feed)

        requestJson = json.dumps(requestMessage)
        self.ws.send(requestJson)

    def __request_challenge(self):
        """Request a challenge from Crypto Facilities Ltd"""

        requestMessage = {
            "event": "challenge",
            "api_key": self.apiKey
        }

        requestJson = json.dumps(requestMessage)
        self.ws.send(requestJson)

    def __wait_for_challenge_auth(self):
        self.__request_challenge()

        print("waiting for challenge...")
        while not self.challengeReady:
            sleep(1)

    def __connect(self):
        self.ws = websocket.WebSocketApp(self.base_url,
                                         on_message=self.__on_message,
                                         on_open=self.__on_open,
                                         on_close=self.__on_close,
                                         on_error=self.__on_error)

        self.wst = Thread(target=lambda: self.ws.run_forever(ping_interval=30))
        self.wst.daemon = True
        self.wst.start()

        conn_timeout = self.timeout
        while (not self.ws.sock or not self.ws.sock.connected) and conn_timeout:
            sleep(1)
            conn_timeout-=1

        if not conn_timeout:
            print("Couldn't connect to: %s",self.baseUrl)
            sys.exit(1)
    def __on_open(self):
        print("Connecting to: %s",self.baseUrl)

    def __on_close(self):
        print("Connection closed")

    def __on_error(self,error):
        print(error)

    def __on_message(self, message):
        messageJson = json.loads(message)

        print(messageJson)

        if messageJson.get("event","") == "challenge":
            self.originalChallenge = messageJson["message"]
            self.signedChallenge = self.__sign_challenge(self.originalChallenge)
            self.challengeReady = True

    def __sign_challenge(self,challenge):

        """Hash the challenge with the SHA-256 algorithm"""

        sha256Hash = hashlib.sha256()
        sha256Hash.update(challenge.encode("utf8"))
        hashDigest = sha256Hash.digest()

        """Base64-decode your api_secret"""
        privateDecoded = base64.b64decode(self.privateKey)

        """Use the result of step 2 to hash the result of step 1 with the HMAC-SHA-512 algorithm"""
        hmacDigest = hmac.new(privateDecoded, hashDigest, hashlib.sha512).digest()

        """Base64-encode the result of step 3"""
        sch = base64.b64encode(hmacDigest).decode("utf-8")

        return sch

print("done")
