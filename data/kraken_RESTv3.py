import time
import base64
import hashlib
import hmac
import json
import urllib.request as urllib2
import urllib.parse as urllib
import ssl


class KrakenApiMethods():
    """rest api Connector"""

    def __init__(self,baseUrl,apiKey="",privateKey="",timeout=10):
        self.baseUrl = baseUrl
        self.apiKey = apiKey
        self.privateKey = privateKey
        self.timeout = timeout

    def get_tickers(self):
        endpoint = "/tickers"
        return self.make_request("GET",endpoint)

    def sign_message(self, endpoint, postData):
        # step 1: concatenate postData, nonce + endpoint
        message = postData + endpoint

        # step 2: hash the result of step 1 with SHA256
        sha256_hash = hashlib.sha256()
        sha256_hash.update(message.encode('utf8'))
        hash_digest = sha256_hash.digest()

        # step 3: base64 decode apiPrivateKey
        secretDecoded = base64.b64decode(self.privateKey)

        # step 4: use result of step 3 to has the result of step 2 with HMAC-SHA512
        hmac_digest = hmac.new(secretDecoded, hash_digest, hashlib.sha512).digest()

        # step 5: base64 encode the result of step 4 and return
        return base64.b64encode(hmac_digest)

    def make_request(self, requestType, endpoint, postUrl="", postBody=""):
        # create authentication headers
        postData = postUrl + postBody

        signature = self.sign_message(endpoint, postData)
        authentHeaders = {"APIKey": self.apiKey, "Authent": signature}

        # create request
        url = self.baseUrl + endpoint + "?" + postUrl
        request = urllib2.Request(url, str.encode(postBody), authentHeaders)
        request.get_method = lambda: requestType


        response = urllib2.urlopen(request, timeout=self.timeout)
        response = response.read().decode("utf-8")

        # return
        return response
