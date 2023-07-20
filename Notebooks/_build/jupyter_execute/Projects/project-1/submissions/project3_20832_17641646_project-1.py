#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#use sample and axis 1. 100 portfolios with 2 stocks, 100 with 5, etc. It is to help you get a nice scatterplot


# In[2]:


pd.set_option('display.float_format', '{:.4f}'.format)
get_ipython().run_line_magic('precision', '4')
plt.rcParams['figure.dpi'] = 100


# In[3]:


import yfinance as yf
import requests_cache
session = requests_cache.CachedSession(expire_after='1D')


# # Calculate daily returns for the S&P 100 stocks.

# In[4]:


url = 'https://en.wikipedia.org/wiki/S%26P_100'
stock_names = pd.read_html(url)
stocks = stock_names[2]['Symbol'].to_list()
#wiki[1].sample(3, axis = 1)


# In[5]:


stocks.remove('BRK.B')


# In[6]:


stocks.append('BRK-B')


# In[7]:


stock_prices = yf.download(stocks, start='2020-01-01', end='2022-09-01', session=session)


# In[8]:


daily_returns = stock_prices['Adj Close'].pct_change()
daily_returns


# # How well do annualized average returns in 2020 predict those in 2021?

# In[9]:


x = daily_returns.loc['2020'].mean().mul(252),
y = daily_returns.loc['2021'].mean().mul(252)

plt.xlabel('Returns in 2020')
plt.ylabel('Returns in 2021')
plt.title('Stock results in 2020 vs. 2021')

plt.scatter(x, y)
plt.show()


# # How well do annualized standard deviations of returns in 2020 predict those in 2021?

# In[10]:


a = daily_returns['2020'].std().mul(np.sqrt(252))
b = daily_returns['2021'].std().mul(np.sqrt(252))

plt.xlabel('StDev in 2020')
plt.ylabel('StDev in 2021')
plt.title('StDev results in 2020 vs. 2021')

plt.scatter(a, b)
plt.show()


# # What are the mean, median, minimum, and maximum pairwise correlations between two stocks?

# In[11]:


correlations = daily_returns.corr()


# In[12]:


mean_per_stock = correlations.mean()
mean_total = mean_per_stock.mean()
'The mean is ' + str(round(mean_total, 4)) + '.'


# In[13]:


median_per_stock = correlations.median()
median_total = median_per_stock.median()
'The median is ' + str(round(median_total, 4)) + '.'


# In[14]:


lowest_correlations = correlations.unstack().sort_values()
min_corr = lowest_correlations.iloc[0]
'The minimum pairwise correlation is ' + str(round(min_corr, 4)) + '.'


# In[15]:


highest_correlations_no_same = correlations.unstack().sort_values().drop_duplicates().iloc[:-1]
max_corr = highest_correlations_no_same.iloc[-1]
'The maximimum pairwise correlation is ' + str(round(max_corr, 4)) + '.'


# The outliers in this analysis are the minimum and maximum. The minimum outlier stems from a significant difference in the stocks analyzed. This is likely a result of a difference in industries as technology companies will experience highly volatile and perhaps large returns while more stable sectors like industrials will experience much less volatile and more market consistent returns. Additionally, the maximum outlier is a result of the exact same companies but different class of stock being analyzed. For example, Google has a Class A (GOOGL) and Class C (GOOG) stock. Therefore, while two different stocks are being analyzed, the returns for Google on their Class A and C will be virtually the exact same.  

# # Plot annualized average returns versus annualized standard deviations of returns.
# 

# In[16]:


# creates a plot given dataframe and variables
def create_plot(plot_data, plot_kind, x_var, y_var, labels=False, trendline=False, **other_vars):
    plot = plot_data.plot(kind=plot_kind, x=x_var, y=y_var, **other_vars)
    # labeling to identify outliers
    if labels:
        for idx, r in plot_data.iterrows():
            plot.annotate(f'${idx}', (r[x_var],r[y_var]))
    # trendline for identifying outliers
    if trendline:
        m,b = np.polyfit(x=plot_data.loc[:, x_var], y=plot_data.loc[:, y_var], deg=1)
        plt.axline(xy1=(0, b), slope=m)
    plt.show()
    return plot

# creates plot of returns vs stdevs when given returns as a dataframe
def create_return_stdev_plot(daily_returns, **other_vars):
    annual_returns = daily_returns.mean().mul(252)
    annual_stdevs = daily_returns.std().mul(np.sqrt(252))
    stock_plot_data = pd.concat([annual_returns.rename('return').mul(100), annual_stdevs.rename('stdev').mul(100)], axis=1)
    create_plot(stock_plot_data, 'scatter', 'stdev', 'return', 
            title='Annualized Average Returns vs Annualized Standard Deviation of Returns', 
            xlabel='Standard Deviation (%)', ylabel='Return (%)', **other_vars)


# In[17]:


create_return_stdev_plot(daily_returns)


# In[18]:


plot = create_return_stdev_plot(daily_returns, labels=True, trendline=True)


# Here, we observe the largest outlier is Tesla ($TSLA), which although containing a high standard deviation, still overperformed its predicted** performance by a relatively large margin of approximately 80%.
# 
# On the flip side, we observe the outlier of Boeing ($BA). While similar to Tesla in having a high standard deviation, Boeing underperformed its predicted** return by about 35%.
# 
# ** prediction via linear regression

# # Repeat the exercise above (question 5) with 100 random portfolios of 2, 5, 10, and 25 stocks.

# In[19]:


# generate a given number of portfolios of given number of stocks from given data
def generate_portfolios(stock_data, num_stocks, num_portfolios=100):
    # 100 random portfolios of 2 stocks
    np.random.seed(15)
    portfolio_returns = pd.DataFrame({}, index=stock_data.index)
    portfolios = []
    while len(portfolios) <= num_portfolios:
        # get random portfolios
        sample_returns = stock_data.sample(num_stocks, axis=1)
        # Guarantee unique samples
        if set(sample_returns.columns) in portfolios:
            break
        portfolios.append(set(sample_returns.columns))
        # Get portfolio returns, add to dataframe 
        portfolio_returns[str(set(sample_returns.columns))] = sample_returns.mean(axis=1)
    return portfolio_returns


# In[20]:


# plot 100 random portfolios of 2 stocks
create_return_stdev_plot(generate_portfolios(daily_returns, 2))


# In[21]:


# plot 100 random portfolios of 5 stocks
create_return_stdev_plot(generate_portfolios(daily_returns, 5))


# In[22]:


# plot 100 random portfolios of 10 stocks
create_return_stdev_plot(generate_portfolios(daily_returns, 10))


# In[23]:


# plot 100 random portfolios of 25 stocks
create_return_stdev_plot(generate_portfolios(daily_returns, 25))

