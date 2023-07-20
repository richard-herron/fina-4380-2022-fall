#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


pd.set_option('display.float_format', '{:.4f}'.format)
get_ipython().run_line_magic('precision', '4')
plt.rcParams['figure.dpi'] = 150


# In[3]:


import yfinance as yf
import requests_cache
session = requests_cache.CachedSession(expire_after='1D')


# # Calculate daily returns for the S&P 100 stocks.

# # How well do annualized average returns in 2020 predict those in 2021?

# # How well do annualized standard deviations of returns in 2020 predict those in 2021?

# # What are the mean, median, minimum, and maximum pairwise correlations between two stocks?

# # Plot annualized average returns versus annualized standard deviations of returns.

# # Repeat the exercise above (question 5) with 100 random portfolios of 2, 5, 10, and 25 stocks.
