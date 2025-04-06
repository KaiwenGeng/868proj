try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen

import certifi
import json

def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)
api_key = "7jLb6gpseT5MzWLfY7S2K1drPwLUWFQ5"
url = ("https://financialmodelingprep.com/stable/historical-chart/1min?symbol=EURUSD&apikey=" + api_key)
print(get_jsonparsed_data(url))