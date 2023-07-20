import time
import pandas as pd
import yfinance as yf

print('My timezone: ' + ' '.join(time.tzname))
df = yf.download('TSLA')
print(df.head())
print(df.tail())