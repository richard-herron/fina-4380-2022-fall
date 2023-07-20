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

# In[4]:


returns = pd.read_html("https://en.wikipedia.org/wiki/S%26P_100#Components")[2]


# In[5]:


ticker = list(returns.loc[:, "Symbol"])

for i in range(len(ticker)):
    ticker[i] = ticker[i].replace('.', '-')


# In[6]:


def download(**kwargs):
    _stocks = yf.download(**kwargs)
    _returns = _stocks["Adj Close"].pct_change()
    _multi_index = pd.MultiIndex.from_product([["Returns"],_stocks["Adj Close"].columns])
    _stocks[_multi_index] = _returns
    return _stocks


# In[7]:


df = download(tickers = ticker, session = session).stack().swaplevel().sort_index()


# In[8]:


returns_daily = pd.DataFrame(df.loc[:,"Returns"]).unstack().unstack().unstack().swaplevel().loc['2020-01-01':'2022-08-31'].swaplevel().unstack().unstack().unstack()
returns_daily


# # How well do annualized average returns in 2020 predict those in 2021?

# In[9]:


print("The annualized average returns in 2020 do not predict those in 2021 very well. The differences between both years can not only be significant, but are not consistent between the different stocks. Some returns are half as small in 2021 compared to 2020, while some stay relatively the same.")
unstack_2020 = returns_daily.unstack().unstack().unstack().swaplevel().loc['2020-01-01':'2020-12-31']
unstack_2021 = returns_daily.unstack().unstack().unstack().swaplevel().loc['2021-01-01':'2021-12-31']
unstack_all = returns_daily.unstack().unstack().unstack().swaplevel()

returns_2020 = pd.DataFrame(unstack_2020.mean().mul(252))
returns_2021 = pd.DataFrame(unstack_2021.mean().mul(252))

returns_2020.columns = ['Annualized Average Returns for 2020']
returns_2021.columns = ['Annualized Average Returns for 2021']

pd.concat([returns_2020,returns_2021], axis = 1)



# # How well do annualized standard deviations of returns in 2020 predict those in 2021?

# In[10]:


print("Similar to the annualized average returns, annualized standard deviations of returns in 2020 do not predict those in 2021 very well. The differences between both years are significant throughout all the stocks.")
sd_2020 = pd.DataFrame(unstack_2020.std().mul(np.sqrt(252)))
sd_2021 = pd.DataFrame(unstack_2021.std().mul(np.sqrt(252)))

sd_2020.columns = ['Annualized Standard Deviations for 2020']
sd_2021.columns = ['Annualized Standard Deviations for 2021']

pd.concat([sd_2020,sd_2021], axis = 1)


# # What are the mean, median, minimum, and maximum pairwise correlations between two stocks?

# In[11]:


corr = unstack_all.corr()
corr2 = unstack_all.corr()
np.fill_diagonal(corr2.values, np.nan)


# In[12]:


corr3 = corr.agg(["mean", "median", "min"]).unstack().unstack()
corr4 = corr2.agg(["max"]).unstack().unstack()
corr3["max"] = corr4["max"]


# In[13]:


corr3


# In[14]:


print("The two lowest outliers for the correlations are between Verizon and Tesla. These companies are very different in both product and in volatility. Verizon is a steady electric telephone company, while Tesla is an electric car company that is constantly in the news. The next two lowest correlations, the Simon Property Group and Thermo Fisher Scientific, are very different in the service they provide. Simon Property Group is a real estate investment trust, where as Thermo Fisher Scientific actually supplies products. The third lowest correlations are also between companies that serve very different demographics. Altria Group is a tobacco company that serves an older demographic, while Netflix is an entertainment company that serves a younger demographic.")
outliers_min = corr3.sort_values("min").head(6)
outliers_min[["min"]]


# In[15]:


print("The two highest outliers for the correlations are between GOOG and GOOGL as they are simply different classes of Aplhabet Inc. The next two outliers are between Bank of America and JP Morgan, both major United States investment major banks. The following two largest outliers are Visa and MasterCard, both major credit card companies.")
outliers_max = corr3.sort_values("max", ascending = False).head(6)
outliers_max[["max"]]


# # Plot annualized average returns versus annualized standard deviations of returns.

# In[16]:


returns_all = pd.DataFrame(unstack_all.mean().mul(252)).mul(100)
sd_all = pd.DataFrame(unstack_all.std().mul(np.sqrt(252))).mul(100)


# In[17]:


plt.scatter(sd_all, returns_all)
plt.xlabel("Standard Deviation (%)")
plt.ylabel("Returns (%)")
plt.title("Returns vs Standard Deviation Jan 2020 - Aug 2022")


# In[18]:


print("The largest outliers for returns were Tesla, NVIDIA, Eli Lilly and Co, Advanced Micro Devices, and ConocoPhillips, which are large technology, pharmaceutical and oil companies that have been performing well.")
outliers_returns_max = returns_all.sort_values(0, ascending = False)
outliers_returns_max.columns=["Top Returns"]
outliers_returns_max.head(5)


# In[19]:


print("the largest outliers for standard deviation were for Tesla, Boeing Co, Simon Propery Group Inc,NVIDIA, and Advanced Micro Devices, which are large technology, airline and real estate companies with large volatilities based on the economy. ")
outliers_sd_max = sd_all.sort_values(0, ascending = False)
outliers_sd_max.columns=["Top Standard Deviation"]
outliers_sd_max.head(5)


# In[20]:


print("The lowest outliers for returns was for Intel, Walgreens Boots Alliance Inc, Verizon, Boeing Co, and AT&T, all companies that have not been performing well.")
outliers_returns_min = returns_all.sort_values(0)
outliers_returns_min.columns=["Bottom Returns"]
outliers_returns_min.head(5)


# In[21]:


print("The lowest outliers for standard deviation were for Verizon,Johnson and Johnson, Bristol-Myers Squibb Co, Colgate, and Proctor and Gamble, all companies with low volatility relative to the performance of the economy.")
outliers_sd_min = sd_all.sort_values(0)
outliers_sd_min.columns=["Bottom Standard Deviation"]
outliers_sd_min.head(5)


# # Repeat the exercise above (question 5) with 100 random portfolios of 2, 5, 10, and 25 stocks.

# In[22]:


np.random.seed(42)


# In[23]:


sample_2_returns = []
for i in range(100):
    sample_2_returns.append(pd.DataFrame(unstack_all.unstack().unstack().unstack().sample(2).mean(axis = 1).mul(252)).mul(100))


# In[24]:


sample_2_sd = []
for i in range(100):
    sample_2_sd.append(pd.DataFrame(unstack_all.unstack().unstack().unstack().sample(2).std(axis = 1).mul(np.sqrt(252))).mul(100))


# In[25]:


plt.scatter(sample_2_sd, sample_2_returns)
plt.xlabel("Standard Deviation (%)")
plt.ylabel("Returns (%)")
plt.title("Returns vs Standard Deviation 100 2-Ticker Portfolios Jan 2020 - Aug 2022")


# In[26]:


sample_5_returns = []
for i in range(100):
    sample_5_returns.append(pd.DataFrame(unstack_all.unstack().unstack().unstack().sample(5).mean(axis=1).mul(252)).mul(100))


# In[27]:


sample_5_sd = []
for i in range(100):
    sample_5_sd.append(pd.DataFrame(unstack_all.unstack().unstack().unstack().sample(5).std(axis = 1).mul(np.sqrt(252))).mul(100))


# In[28]:


plt.scatter(sample_5_sd, sample_5_returns)
plt.xlabel("Standard Deviation (%)")
plt.ylabel("Returns (%)")
plt.title("Returns vs Standard Deviation 100 5-Ticker Portfolios Jan 2020 - Aug 2022")


# In[29]:


sample_10_returns = []
for i in range(100):
    sample_10_returns.append(pd.DataFrame(unstack_all.unstack().unstack().unstack().sample(10).mean(axis=1).mul(252)).mul(100))


# In[30]:


sample_10_sd = []
for i in range(100):
    sample_10_sd.append(pd.DataFrame(unstack_all.unstack().unstack().unstack().sample(10).std(axis = 1).mul(np.sqrt(252))).mul(100))


# In[31]:


plt.scatter(sample_10_sd, sample_10_returns)
plt.xlabel("Standard Deviation (%)")
plt.ylabel("Returns (%)")
plt.title("Returns vs Standard Deviation 100 10-Ticker Portfolios Jan 2020 - Aug 2022")


# In[32]:


sample_25_returns = []
for i in range(100):
    sample_25_returns.append(pd.DataFrame(unstack_all.unstack().unstack().unstack().sample(25).mean(axis=1).mul(252)).mul(100))


# In[33]:


sample_25_sd = []
for i in range(100):
    sample_25_sd.append(pd.DataFrame(unstack_all.unstack().unstack().unstack().sample(25).std(axis = 1).mul(np.sqrt(252))).mul(100))


# In[34]:


plt.scatter(sample_25_sd, sample_25_returns)
plt.xlabel("Standard Deviation (%)")
plt.ylabel("Returns (%)")
plt.title("Returns vs Standard Deviation 100 25-Ticker Portfolios Jan 2020 - Aug 2022")

