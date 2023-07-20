#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os


# In[2]:


pd.set_option('display.float_format', '{:.4f}'.format)
get_ipython().run_line_magic('precision', '4')
plt.rcParams['figure.dpi'] = 150


# In[3]:


from datetime import timedelta
import yfinance as yf
import requests_cache
session = requests_cache.CachedSession(expire_after=timedelta(days=1))


# # Calculate daily returns for the S&P 100 stocks.

# In[4]:


sp_assets = pd.read_html(
        'https://en.wikipedia.org/wiki/S%26P_100#Components')[2]
tickers = sp_assets['Symbol'].str.replace('.', '-').tolist()
SAP_100_2020 = yf.download(tickers=tickers, session=session, start='2020-01-01', end='2022-08-01')
returns_2020 = SAP_100_2020['Adj Close'].pct_change() * 100


# # How well do annualized average returns in 2020 predict those in 2021?

# In[5]:


SAP_100_2021 = yf.download(tickers=tickers, session=session, start='2021-01-01',
                 end='2021-08-01')
returns_2021 = SAP_100_2021['Adj Close'].pct_change() * 100
ann_avg_return_2020 = returns_2020.mean() * 252
ann_avg_return_2021 = returns_2021.mean() * 252


# In[6]:


avg_corr = ann_avg_return_2020.corr(ann_avg_return_2021) * 100
print(f"We found that the annualized average returns in 2020 did not accuratley predicted those in 2021 due to the correlation being{avg_corr: .4f}%")


# # How well do annualized standard deviations of returns in 2020 predict those in 2021?

# In[7]:


ann_std_return_2020 = returns_2020.std() * math.sqrt(252)
ann_std_return_2021 = returns_2021.std() * math.sqrt(252)


# In[8]:


std_corr = ann_std_return_2020.corr(ann_std_return_2021) * 100
print(f"We found that the annualized standard deviations of returns in 2020 predicted those in 2021 well due to the correlation being{std_corr: .4f}%") 


# # What are the mean, median, minimum, and maximum pairwise correlations between two stocks?

# In[9]:


corrs


# In[9]:


# get low triangle elements of correlation matrix
corrs = np.tril(returns_2020.corr(), -1)[np.tril(returns_2020.corr(), -1) != 0]
corrs = corrs * 100


# In[10]:


# find the outliners of corrs
outliers = corrs[(corrs - corrs.mean()) > 2 * np.std(corrs)]


# In[11]:


# remove top 5% and bottom 5% data
corrs_sorted = np.sort(corrs)
pruned_corrs = corrs_sorted[math.floor(len(corrs) * 0.05) : math.floor(len(corrs) * 0.95) ]


# In[12]:


# compute mean, median, minimum, and maximum
print(f"The minimum of pruned pairwise correlations: {pruned_corrs.min():.4f}%")
print(f"The maximum of pruned pairwise correlations: {pruned_corrs.max():.4f}%")
print(f"The median of pruned pairwise correlations: {np.median(pruned_corrs):.4f}%", )
print(f"The mean of pruned pairwise correlations: {pruned_corrs.mean():.4f}%")


# # Plot annualized average returns versus annualized standard deviations of returns.

# In[13]:


# Create a scatter plot with labels and a title
x = ann_avg_return_2020
y = ann_std_return_2020
plt.scatter(x, y, c='red')
plt.xlabel('Annualized Average Returns (Percent)')
plt.ylabel('Annualized Standard Deviations (Percent)')
plt.title('Annualized Average Returns vs Standard Deviations of Returns')

# calculate the trendline
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")


# ## Discussion on Outliers:
# In this graph, we can see that there are some outliers, notably one at approximately (120,75) and two near (0, 65). For the first outlier, we see that the mean return is quite high as well as the standard deviation. This demonstrates that this stock generally returns in a very volatile manner, with the volatility ocurring around a 120% return. For the second set of outliers, these stocks return around a mean return of 0% with a high standard deviation or volatility. This means that there is strong possibility that the return could be either positive or negative. 

# # Repeat the exercise above (question 5) with 100 random portfolios of 2, 5, 10, and 25 stocks.

# In[14]:


def portfolio(a): #method for simplifying the call of 100 random portfolios with varying amounts of stocks denoted as var a
    for x in range(100): #100 random portfolios 
        df = pd.DataFrame(returns_2020) #storing sp100 stocks to a df
        df = df.sample(n=a,axis='columns') #selecting 'a' amount of random stocks from the df
        df = df.dropna() #dropping rows with null values 
        ann_pre_avg2_return = df.mean(axis=0) *252 #mean calculation
        ann_pre_std2_return = df.std(axis=0) * math.sqrt(252) #std calculation 
        ann_avg2_return = ann_pre_avg2_return.mean() #aggregated mean
        ann_std2_return = ann_pre_std2_return.mean() #aggregated std 
        plt.scatter(ann_avg2_return, ann_std2_return, c='red') #scatterplot
    #Creates the labels and titles of the graph
    plt.xlabel('Annualized Average Returns 2020 (Percent)')
    plt.ylabel('Annualized Standard Deviations of Returns 2020 (Percent)')
    plt.title('Annualized Average Returns vs Standard Deviations of Returns (100 portfolios of ' +str(a)+' random stocks)')
    #Defines the limits of the plots
    plt.xlim(-5, 60)
    plt.ylim(20,70)


# In[15]:


portfolio(2)


# In[16]:


portfolio(5)


# In[17]:


portfolio(10)


# In[18]:


portfolio(25)


# # Question 6 Discussion
# As we can see per the graphs above, it appears that as we increase the number of different stocks, both the average returns and the standard deviations become much more congested. This makes sense, as the more diversification of ones portfolio should lead to a lower standard deviation for the overall portfolio which would also lead to a more clustered average return.
