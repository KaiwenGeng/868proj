#!/usr/bin/env python
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen
import certifi
import json
import re  # For simple HTML tag removal

def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

def extract_clean_text(html_content):
    # Remove HTML tags with regex
    clean_text = re.sub(r'<.*?>', '', html_content)
    return clean_text

def get_fmp_news(api_key):   # Latest General News
    url = "https://financialmodelingprep.com/stable/fmp-articles?apikey=" + api_key 
    data = get_jsonparsed_data(url)
    date_content_pairs = []
    for article in data:
        date = article["date"]
        clean_content = extract_clean_text(article["content"])
        date_content_pairs.append((date, clean_content))
    return date_content_pairs

def get_forex_news(api_key, currency_pair): # Latest Forex News
    url = "https://financialmodelingprep.com/stable/news/forex?symbols=" + currency_pair + "&apikey=" + api_key
    data = get_jsonparsed_data(url)
    date_content_pairs = []
    for article in data:
        date = article["publishedDate"]
        clean_content = extract_clean_text(article["text"])
        date_content_pairs.append((date, clean_content))
    return date_content_pairs
    

   