#!/usr/bin/env python
# coding: utf-8

# # Herron - Downloading Data
# 
# This notebook shows how to use the yfinance, pandas-datareader, and requests-cache packages to download data from 
#     [Yahoo! Finance](https://finance.yahoo.com/), 
#     [the Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html), 
#     [the Federal Reserve Economic Database (FRED)](https://fred.stlouisfed.org/), 
#     and others.
# For completeness, this notebooks also covers saving to and reading from .csv and .pkl files.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('precision', '4')
pd.options.display.float_format = '{:.4f}'.format


# ## The yfinance Package

# The [yfinance package](https://github.com/ranaroussi/yfinance) provides "a reliable, threaded, and Pythonic way to download historical market data from Yahoo! finance."
# Other packages that provide similar functionality, but I think yfinance is simplest to use.
# To avoid repeated calls to Yahoo! Finance's advanced programming interface (API), we will use the [requests-cache package](https://github.com/requests-cache/requests-cache).
# These packages should already be installed in your DataCamp Workspace environment.
# If not, we can install these packages with the `%pip` magic in the following cell, which we only need to run once.
# If you use a local installation of the Anaconda distribution, you can instead run `! conda install -y -c conda-forge yfinance requests-cache`.

# In[3]:


# %pip install yfinance requests-cache


# In[4]:


# ! conda install -y -c conda-forge yfinance requests-cache


# In[5]:


import yfinance as yf
import requests_cache
session = requests_cache.CachedSession(expire_after='1D')


# We can download data for the FAANG stocks (Facebook, Amazon, Apple, Netflix, and Google).
# We can pass tickers as either a space-delimited string or a list of strings.

# In[6]:


faang = yf.download(tickers='META AMZN AAPL NFLX GOOG', session=session)


# In[7]:


( # we can insert line breaks inside the chain if we wrap the chain in ()
    faang # start with the faang data frame
    ['Adj Close'] # grab all adjusted closes
    .pct_change() # calculate the percent change, which is a return that accounts for splits and dividends
    .loc['2022'] # select returns from 2022
    .add(1) # add 1
    .cumprod() # compound returns
    .sub(1) # subtract 1 to get cumulative (year-to-date) returns
    .mul(100) # multiply by 100 to convert decimal returns to percent returns
    .plot() # plot
)
plt.ylabel('Year-to-Date Return (%)') # add label to y axis (vertical axis)
plt.title('Year-to-Date Returns for FAANG Stocks') # add title
plt.show() # suppress output from plt.title()


# ## The pandas-datareader package

# The [pandas-datareader](https://pandas-datareader.readthedocs.io/en/latest/index.html) package provides easy access to a variety of data sources, such as 
#     [the Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) 
#     and 
#     [the Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/).
# The pandas-datareader package also provides access to Yahoo! Finance data, but the yfinance package has better documentation.
# The pandas-datareader packages should already be installed in your DataCamp Workspace environment.
# If not, we can install these packages with the `%pip` magic in the following cell, which we only need to run once.
# If you use a local installation of the Anaconda distribution, you can instead run `! conda install -y -c conda-forge pandas-datareader`.

# In[8]:


# %pip install pandas-datareader


# In[9]:


# ! conda install -y -c conda-forge pandas-datareader


# We will use `pdr` as the abbreviated prefix for pandas-datareader.

# In[10]:


import pandas_datareader as pdr


# He is an example with the daily benchmark factor from Ken French's Data Library.
# The `get_available_datasets()` function provides the exact names for all of Ken French's data sets.

# In[11]:


pdr.famafrench.get_available_datasets(session=session)[:5]


# Note that pandas-datareader returns a dictionary of data frames and returns the most recent five years of data unless we specify a `start` date.
# Most of French's data are available back through the second half od 1926.

# In[12]:


ff = pdr.get_data_famafrench('F-F_Research_Data_Factors_daily', start='1900', session=session)


# In[13]:


ff[0]


# In[14]:


print(ff['DESCR'])


# ## Saving and Reading Data with .csv and .pkl Files

# The universal way to save data is to a .csv file (i.e., a file with comma-separated values) with the `.to_csv()` method.
# You may need to add a "Data" folder at the same hieracrchy as your "Notebooks" folder using the "File Browser" in JupyterLab's left sidebar.

# In[15]:


faang.to_csv('../../Data/FAANG.csv')


# We have to pass several arguments to `pd.read_csv()` since the `faang` data frame has a column multiindex (i.e., one level of variables and another for tickers).

# In[16]:


pd.read_csv('../../Data/FAANG.csv', header=[0,1], index_col=[0], parse_dates=True)


# We can use a .pkl file to save and read a pandas object as-is.
# These .pkl files are easier to use than .csv files but less universal.

# In[17]:


faang.to_pickle('../../Data/FAANG.pkl')


# In[18]:


pd.read_pickle('../../Data/FAANG.pkl')

