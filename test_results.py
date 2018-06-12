import requests
import config
import pandas as pd

url = 'https://www.quandl.com/api/v3/datasets/ZILLOW/C25709_ZRISFRR?api_key='+config.api_key

r = requests.get(url)

json_data = r.json()

df = pd.DataFrame(json_data)