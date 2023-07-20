#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore


# In[2]:


pd.set_option('display.float_format', '{:.4f}'.format)
get_ipython().run_line_magic('precision', '4')
plt.rcParams['figure.dpi'] = 150


# In[3]:


import yfinance as yf
import requests_cache
session = requests_cache.CachedSession


# In[4]:


requests_cache.CachedSession


# In[5]:


requests_cache.CachedSession()


# # Calculate daily returns for the S&P 100 stocks. 
# Use all the stocks listed here: https://en.wikipedia.org/wiki/S%26P_100#Components

# In[6]:


# extract table of stocks from Wikipedia provided
sp_100_companies = pd.read_html('https://en.wikipedia.org/wiki/S%26P_100#Components')
sp_100_companies = sp_100_companies[2]
sp_100_companies


# In[7]:


# replace BRK.B with BRK-B
sp_100_companies.iloc[18].loc['Symbol'] = 'BRK-B'


# In[8]:


# create list of stocks to download
stocks = list(sp_100_companies.loc[:, 'Symbol'])

# request past daily stock prices
yf_data = yf.download(stocks, start='2020-01-01', end='2022-08-31', session=session)
yf_data = pd.DataFrame(yf_data)
yf_data.head()


# In[9]:


yf_data = yf.download(stocks, start='2020-01-01', end='2022-08-31', session=session)


# In[10]:


yf_data


# In[11]:


# make list of stocks for future reference
stocks = list()

for tuple in yf_data.keys()[:101]:
    stocks.append(tuple[1])


# In[12]:


# define function that calculates a daily return for a single stock across an open and a close - all resulting values are in %
def daily_return(open, close):
    return (close - open) / open * 100


# In[13]:


# define function that calculates average daily returns of a single stock - all resulting values are in %
def avg_return(stock, data = yf_data):
    # get all adj close values for the provided stock
    stock_adj_close = data[('Adj Close', stock)]
    # iterate through each day except for the last day
    all_days_returns = list()
    for index, day in enumerate(stock_adj_close.values[:-1]):
        all_days_returns.append(daily_return(day, stock_adj_close[index + 1]))
    return all_days_returns


# In[14]:


# create data frame of each stock's daily returns
def calculate_daily_returns(stocks=stocks):
    df = pd.DataFrame()
    for stock in stocks:
        df[stock] = avg_return(stock)
    return df


# In[15]:


# calculate and display dataframe of each stocks daily returns
df_daily_returns = calculate_daily_returns()
df_daily_returns['Date'] = yf_data.index[:-1]
df_daily_returns = df_daily_returns.set_index('Date')
df_daily_returns.head()


# # How well do annualized average returns in 2020 predict those in 2021?
# Annualize the mean of daily returns by multiplying by 252.

# In[16]:


# calculates average returns of each stock only from the year 2020
df_avg_returns_2020 = df_daily_returns.loc[:'2020-12-31']

# calculate the annualized average returns from 2020
df_annualized_avg_returns_2020 = df_avg_returns_2020.mean() * 252
df_annualized_avg_returns_2020


# In[17]:


# calculates average returns of each stock only from the year 2021
df_avg_returns_2021 = df_daily_returns.loc['2021-01-01':'2021-12-31']

# calculate the annualized average returns from 2021
df_annualized_avg_returns_2021 = df_avg_returns_2021.mean() * 252
df_annualized_avg_returns_2021


# In[18]:


plt.figure(figsize=(5, 3))

plt.scatter(df_annualized_avg_returns_2020, df_annualized_avg_returns_2021, s=3)
plt.title('Annualized Average Daily Returns of S&P100 Stocks Between 2020 and 2021')
plt.xlabel('2020 Annualized Average Daily Returns')
plt.ylabel('2021 Ann. Avg. Daily Returns')
plt.show()


# In[19]:


df_annualized_avg_returns_2020.corr(df_annualized_avg_returns_2021)


# The annualized average daily return of the S&P100 had a small correlation between 2020 and 2021. Therefore the annualized average return of 2020 would not be a good prediction of 2021. We should consider however that 2020 was highly impacted by Covid-19, therefore using this method for different years might have a higher correlation. 

# # How well do annualized standard deviations of returns in 2020 predict those in 2021?
# 
# Annualize the standard deviation of daily returns by multiplying by the square root of 252.

# In[20]:


# calculates standard deviations of each stock only from the year 2020
df_std_dev_2020 = df_daily_returns.loc[:'2020-12-31'].std()

# calculate the annualized standard deviations of returns from 2020
df_annualized_std_2020 = df_std_dev_2020 * np.sqrt(252)
df_annualized_std_2020 


# In[21]:


# calculates standard deviations of each stock only from the year 2020
df_std_dev_2021 = df_daily_returns.loc['2021-01-01':'2021-12-31'].std()

# calculate the annualized standard deviations of returns from 2020
df_annualized_std_2021 = df_std_dev_2021 * np.sqrt(252)
df_annualized_std_2021 


# In[22]:


plt.figure(figsize=(5, 4.5))

plt.scatter(df_annualized_std_2020, df_annualized_std_2021, s=3)
plt.title('Annualized Standard Deviations of S&P100 Stocks Between 2020 and 2021')
plt.xlabel('2020 Annualized Standard Deviations of Stocks')
plt.ylabel('2021 Annualized Standard Deviations of Stocks')

plt.show()


# In[23]:


df_std_dev_2020.corr(df_std_dev_2021)


# The annualized standard deviation of the S&P 100 had a fairly strong correlation between 2020 and 2021. Annualized standard deviation of 2020 would therefore be a fairly good prediction for those of 2021. 

# # What are the mean, median, minimum, and maximum pairwise correlations between two stocks?
# Discuss and explain any outliers.

# In[24]:


corr_matrix= df_avg_returns_2021.corr()
correlation= corr_matrix.stack().reset_index()
corr_list =np.triu(corr_matrix,k=1)
corr_list[corr_list==0]=np.nan
    
print("Mean pairwise correlation:", np.nanmean (corr_list))
print("Median pairwise correlation:", np.nanmedian (corr_list))
print("Maximum pairwise correlation:", np.nanmax(corr_list))
print("Minimum pairwise correlation:", np.nanmin(corr_list))


# In[25]:


corr_list


# In[26]:


min_pair_corr= df_avg_returns_2021.corr().unstack().sort_values()
print(min_pair_corr)
max_pair_corr= df_avg_returns_2021.corr().unstack().sort_values()
print(max_pair_corr)


# # Plot annualized average returns versus annualized standard deviations of returns.

# In[27]:


df_annualized_avg_returns = df_daily_returns.mean() * 252
df_standard_dev = df_daily_returns.std() * np.sqrt(252)

plt.figure(figsize=(5, 3))
plt.scatter(df_annualized_avg_returns, df_standard_dev, s=3)
plt.xlabel('Annualized Average Returns')
plt.ylabel('Annualized Std.Dev of Returns')
plt.title('Annualized Average Return vs. Annualized Std.Dev of Returns')

m, b = np.polyfit(df_annualized_avg_returns, df_standard_dev, 1)
plt.plot(df_annualized_avg_returns, m*df_annualized_avg_returns+b)
plt.show()

df_annualized_avg_returns.corr(df_standard_dev)


# In[28]:


df_annualized_avg_returns.drop('TSLA')


# In[29]:


m, b = np.polyfit(df_annualized_avg_returns.drop('TSLA'), df_standard_dev.drop('TSLA'), 1)


# In[30]:


plt.scatter(df_annualized_avg_returns.drop('TSLA'), df_standard_dev.drop('TSLA'), s=3)


# ## Analysis/Discussion of Results and Outliers
# The total annualized returns when compared to their respective standard deviations are weakly positively correlated. The coefficient is .349 making it on the higher end of the weak range for correlation. This does not provide any conclusive data since the weak correlation provides only a minor trend and nothing substantial. If we are to contribute the correlation to any factor, looking at the industries for each company can provide insight. Inside the S&P 100, the industry breakdown is weighted towards technology, pharmaceuticals, and financial companies. Intra-industry comparables can marginally account for the such correlation. 
# 

# # Repeat the exercise above (question 5) with 100 random portfolios of 2, 5, 10, and 25 stocks.
# For simplicity, use equal-weighted portfolios and re-balance daily. These portfolio returns are df.mean(axis=1) if data frame df contains columns of daily returns. The .sample() method can randomly sample columns to create random portfolios.

# In[31]:


# input `size` must be 2, 5, 10, or 25 to yield results desired
def get_portfolio(size):
    return df_daily_returns.sample(size, axis=1)


# In[32]:


def get_mean(size):
    return get_portfolio(size).mean(axis=1)


# In[33]:


def get_std(size):
    return get_portfolio(size).std(axis=1)


# In[34]:


# input `size` must be 2, 5, 10, or 25 to yield results desired
def get_portfolio_avg_returns(size):

    # create empty df to store values
    avg_returns = list()
    xlabel = list()

    for i in range(1, 101):
        avg_returns.append(get_mean(size))
        xlabel.append(f'Portfolio {i}')
    
    df = pd.DataFrame(avg_returns).transpose()
    df.columns = xlabel
    
    return df


# In[35]:


# input `size` must be 2, 5, 10, or 25 to yield results desired
def get_portfolio_std(size):

    # create empty df to store values
    avg_returns = list()
    xlabel = list()

    for i in range(1, 101):
        avg_returns.append(get_std(size))
        xlabel.append(f'Portfolio {i}')
    
    df = pd.DataFrame(avg_returns).transpose()
    df.columns = xlabel
    
    return df


# In[36]:


size_two_returns = get_portfolio_avg_returns(2)
size_five_returns = get_portfolio_avg_returns(5)
size_ten_returns = get_portfolio_avg_returns(10)
size_twentyfive_returns = get_portfolio_avg_returns(25)
size_two_std = get_portfolio_std(2)
size_five_std = get_portfolio_std(5)
size_ten_std = get_portfolio_std(10)
size_twentyfive_std = get_portfolio_std(25)


# In[37]:


# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(size_two_returns, size_two_std, alpha=0.05)
axs[0, 0].set_title('Axis [0, 0]')
#axs[0, 1].scatter(size_five_returns, size_five_std, 'tab:orange', alpha=0.05)
#axs[0, 1].set_title('Axis [0, 1]')
#axs[1, 0].scatter(size_ten_returns, size_ten_std, 'tab:green', alpha=0.05)
#axs[1, 0].set_title('Axis [1, 0]')
#axs[1, 1].scatter(size_twentyfive_returns, size_twentyfive_std, 'tab:red',alpha=0.05)
#axs[1, 1].set_title('Axis [1, 1]')

#for ax in axs.flat:
#    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
#for ax in axs.flat:
#    ax.label_outer()

