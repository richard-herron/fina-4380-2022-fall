#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

pd.set_option('display.float_format', '{:.4f}'.format)
get_ipython().run_line_magic('precision', '4')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import pandas_datareader as pdr
import requests_cache
session = requests_cache.CachedSession(expire_after='1D')


# In[2]:


ff = pdr.get_data_famafrench(
    symbols='F-F_Research_Data_Factors',
    start='1900',
    session=session
)


# # Consider only 1999 through today

# # Can you find a period that reverses your question 1 conclusion?

# # Consider all full years from 1927 through 2021

# # Consider all available 20-year holding periods

# # Which investing strategy is better overall, LS or DCA?
