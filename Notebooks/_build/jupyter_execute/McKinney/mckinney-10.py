#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 10 - Data Aggregation and Group Operations

# ## Introduction
# 
# Chapter 10 of Wes McKinney's [*Python for Data Analysis*](https://wesmckinney.com/pages/book.html) discusses groupby operations, which are the pandas equivalent of Excel pivot tables.
# Pivot tables help us calculate statistics (e.g., sum, mean, and median) for one set of variables by groups of other variables (e.g., weekday or year).
# For example, we could use a pivot table to calculate mean daily stock returns by weekday.
# 
# We will focus on:
# 
# 1. Using `.groupby()` to group by columns, indexes, and functions
# 1. Using `.agg()` to aggregate multiple functions
# 1. Using pivot tables as an alternative to `.groupby()`
# 
# ***Note:*** Indented block quotes are from McKinney, and section numbers differ from McKinney because we will not discuss every topic.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('precision', '4')
pd.options.display.float_format = '{:.4f}'.format


# In[3]:


import requests_cache
session = requests_cache.CachedSession(expire_after='1D')
import yfinance as yf
import pandas_datareader as pdr


# ## GroupBy Mechanics
# 
# "Split-apply-combine" is an excellent way to describe and visualize pandas groupby operations.
# 
# > Hadley Wickham, an author of many popular packages for the R programming 
# language, coined the term split-apply-combine for describing group operations. In the
# first stage of the process, data contained in a pandas object, whether a Series, DataFrame, or otherwise, is split into groups based on one or more keys that you provide.
# The splitting is performed on a particular axis of an object. For example, a DataFrame
# can be grouped on its rows (axis=0) or its columns (axis=1). Once this is done, a
# function is applied to each group, producing a new value. Finally, the results of all
# those function applications are combined into a result object. The form of the resulting object will usually depend on what’s being done to the data. See Figure 10-1 for a
# mockup of a simple group aggregation.
# 
# Figure 10-1 visualizes a pandas groupby operation that:
# 
# 1. Splits the dataframe by the `key` column (i.e., "groups by `key`")
# 2. Applies the sum operation to the `data` column (i.e., "sums `data`")
# 3. Combines the grouped sums
# 
# I describe this operation as "sum the `data` column by (gruops formed on) the `key` column."

# In[4]:


np.random.seed(42)
df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                   'key2' : ['one', 'two', 'one', 'two', 'one'],
                   'data1' : np.random.randn(5),
                   'data2' : np.random.randn(5)})


# In[5]:


df


# Here is one way to calculate the means of `data1` by (grouops formed on) `key1`.

# In[6]:


df.loc[df['key1'] == 'a', 'data1'].mean()


# In[7]:


df.loc[df['key1'] == 'b', 'data1'].mean()


# We can do this calculation more quickly:
# 
# 1. Use the `.groupby()` method to group by `key1`
# 2. Use the `.mean()` method to sum `data1` within each value of `key1`
# 
# Note that without the `.mean()` method, pandas only sets up the grouped object, which can accept the `.mean()` method.

# In[8]:


grouped = df['data1'].groupby(df['key1'])


# In[9]:


grouped


# In[10]:


grouped.mean()


# We can can chain the `.groupby()` and `.mean()` methods.

# In[11]:


df['data1'].groupby(df['key1']).mean()


# ---
# ***What does `np.random.randn(5)` do?***
# The funtion `np.random.randn()` creates 5 standard normal random variables (mean of 0 and standard deviation of 1).

# In[12]:


np.random.seed(42)
randos = np.random.randn(1_000_000)
print(f'Mean of {randos.mean():0.4f}, standard deviation of {randos.std():0.4f}')


# ---

# If we prefer our result as a dataframe instead of a series, we can wrap `data1` with two sets of square brackets.

# In[13]:


df[['data1']].groupby(df['key1']).mean()


# We can group by more than one variable.
# We get a hierarchical row index (or row multi-index) when we group by more than one variable.

# In[14]:


means = df['data1'].groupby([df['key1'], df['key2']]).mean()
means


# We can use the `.unstack()` method if we want to use both rows and columns to organize data.
# Note that, the `.unstack()` method un-stacks the last dimension (i.e., `level = -1`) by default so that `key2` values become columns.

# In[15]:


means.unstack()


# The grouping variables can also be columns in the data frame passed to the `.groupby()` method.
# I prefer this approach because we will typically have all data in one data frame.

# In[16]:


df.groupby('key1')[['data1', 'data2']].mean()


# In[17]:


df.groupby(['key1', 'key2']).mean()


# There are many more methods than `.mean()`.
# We can use tab completion to discover (or remind ourselves of) these other methods.

# ### Iterating Over Groups
# 
# We can iterate over groups, too.
# The `.groupby()` method generates a sequence of tuples.
# Each tuples contains the value(s) of the grouping variable(s) and associated chunk of the dataframe.
# McKinney provides two loops to show how to iterate over groups.

# In[18]:


for name, group in df.groupby('key1'):
    print(name)
    print(group)


# In[19]:


for (k1, k2), group in df.groupby(['key1', 'key2']):
    print((k1, k2))
    print(group)


# ### Selecting a Column or Subset of Columns
# 
# We preview the idea of grouping an entire dataframe above.
# However, I want to take this chance to explain McKinney's use of the phrase "syntactic sugar."
# Here is the context:
# 
# > Indexing a GroupBy object created from a DataFrame with a column name or array
# of column names has the effect of column subsetting for aggregation. This means
# that:
# >
# > ```
# > df.groupby('key1')['data1']
# > df.groupby('key1')[['data2']]
# > ```
# >
# > are syntactic sugar for
# >
# > ```
# > df['data1'].groupby(df['key1'])
# > df[['data2']].groupby(df['key1'])
# > ```
# 
# "Syntactic sugar" makes code easier to type or read without adding functionality.
# It makes code "sweeter" for humans to type or read by making it more concise or clear.
# The implication is that syntactic sugar makes code faster to type/read but does make code faster to execute.

# In[20]:


df.groupby(['key1', 'key2'])[['data2']].mean()


# ### Grouping with Functions
# 
# We can also group with functions.
# Below, we group with the `len` function, which calculates the length of the first names in the row index.
# We could instead add a helper column to `people`, but it is easier to pass a function to `.groupby()`.

# In[21]:


np.random.seed(42)
people = pd.DataFrame(
    data=np.random.randn(5, 5), 
    columns=['a', 'b', 'c', 'd', 'e'], 
    index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis']
)
people


# In[22]:


people.groupby(len).sum()


# We can mix functions, lists, etc. that we pass to `.groupby()`.

# In[23]:


key_list = ['one', 'one', 'one', 'two', 'two']
people.groupby([len, key_list]).min()


# In[24]:


d = {'Joe': 'a', 'Jim': 'b'}
people.groupby([len, d]).min()


# In[25]:


d_2 = {'Joe': 'Cool', 'Jim': 'Nerd', 'Travis': 'Cool'}
people.groupby([len, d_2]).min()


# ### Grouping by Index Levels
# 
# We can also group by index levels.
# We can specify index levels by either level number or name.

# In[26]:


columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
                                    [1, 3, 5, 1, 3]],
                                    names=['cty', 'tenor'])
hier_df = pd.DataFrame(np.random.randn(4, 5), columns=columns)


# In[27]:


hier_df.groupby(level='cty', axis=1).count()


# In[28]:


hier_df.groupby(level='cty', axis='columns').count()


# In[29]:


hier_df.groupby(level='tenor', axis=1).count()


# ## Data Aggregation
# 
# Table 10-1 provides the optimized groupby methods:
# 
# - `count`: Number of non-NA values in the group
# - `sum`: Sum of non-NA values
# - `mean`: Mean of non-NA values
# - `median`: Arithmetic median of non-NA values
# - `std`, `var`: Unbiased (n – 1 denominator) standard deviation and variance
# - `min`, `max`: Minimum and maximum of non-NA values
# - `prod`: Product of non-NA values
# - `first`, `last`: First and last non-NA values
# 
# These optimized methods are fast and efficient, but pandas does not limit us to these methods.
# First, any series method is available.

# In[30]:


df.groupby('key1')['data1'].quantile(0.9)


# Second, we can write our own functions and pass them to the `.agg()` method.
# These functions should accept an array and returns a single value.

# In[31]:


def peak_to_peak(arr):
    return arr.max() - arr.min()


# In[32]:


df.groupby('key1')['data1'].agg(peak_to_peak)


# Some other methods work, too, even if they are do not aggregate an array to a single value.

# In[33]:


df.groupby('key1')['data1'].describe()


# ### Column-Wise and Multiple Function Application
# 
# The `.agg()` methods provides two more handy features:
# 
# 1. We can pass multiple functions to operate on all of the columns
# 2. We can pass specific functions to operate on specific columns

# Here is an example with multiple functions:

# In[34]:


df.groupby('key1')['data1'].agg(['mean', 'median', 'min', 'max'])


# In[35]:


df.groupby('key1')[['data1', 'data2']].agg(['mean', 'median', 'min', 'max'])


# What if I wanted to calculate the mean of `data1` and the median of `data2` by `key1`?

# In[36]:


df.groupby('key1').agg({'data1': 'mean', 'data2': 'median'})


# What if I wanted to calculate the mean *and standard deviation* of `data1` and the median of `data2` by `key1`?

# In[37]:


df.groupby('key1').agg({'data1': ['mean', 'std'], 'data2': 'median'})


# ## Apply: General split-apply-combine
# 
# The `.agg()` method aggrates an array to a single value.
# We can use the `.apply()` method for more general calculations.
# 
# We can combine the `.groupby()` and `.apply()` methods to:
# 
# 1. Split a dataframe by grouping variables
# 2. Call the applied function on each chunk of the original dataframe
# 3. Recombine the output of the applied function

# In[38]:


def top(x, col, n=1):
    return x.sort_values(col).head(n)


# In[39]:


df.groupby('key1').apply(top, col='data1', n=2)


# In[40]:


df.groupby('key1').apply(top, col='data2', n=2)


# ## Pivot Tables and Cross-Tabulation
# 
# Above we manually made pivot tables with the `groupby()`, `.agg()`, `.apply()` and `.unstack()` methods.
# pandas provides a literal interpreation of Excel-style pivot tables with the `.pivot_table()` method and the `pandas.pivot_table()` function.
# These also provide row and column totals via "margins".
# It is worthwhile to read-through the `.pivot_table()` docstring several times.

# In[41]:


ind = (
    yf.download(tickers='^GSPC ^DJI ^IXIC ^FTSE ^N225 ^HSI', session=session)
    .rename_axis(columns=['Variable', 'Index'])
    .stack()
)


# The default aggregation function for `.pivot_table()` is `mean`.

# In[42]:


ind.loc['2015':].pivot_table(index='Index')


# We can use 
#     `values` to select specific variables, 
#     `pd.Grouper()` to sample different date windows, 
#     and 
#     `aggfunc` to select specific aggregation functions.

# In[43]:


(
    ind
    .loc['2015':]
    .reset_index()
    .pivot_table(
        values='Close',
        index=pd.Grouper(key='Date', freq='A'),
        columns='Index',
        aggfunc=['min', 'max']
    )
)


# ## Practice

# ***Practice:***
# Calculate the means of columns `data1` and `data2` by `key1` and `key2`, and arrange the results so that values of `key1` are in the rows and values of `key2` are in the columns.

# In[44]:


np.random.seed(42)
df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                   'key2' : ['one', 'two', 'one', 'two', 'one'],
                   'data1' : np.random.randn(5),
                   'data2' : np.random.randn(5)})


# In[45]:


# [['data1', 'data2']] is optional because those are the only remaining columns
practice_1 = df.groupby(['key1', 'key2'])[['data1', 'data2']].mean().unstack()
practice_1


# ***Practice:***
# Replicate the previous practice exercise with `pd.pivot_table()` and test equality with `np.allclose()`.
# We will learn more about `pd.pivot_table()` at the end of this notebook, but we can give it a try now.

# In[46]:


practice_2 = pd.pivot_table(
    data=df,
    values=['data1', 'data2'],
    index='key1',
    columns='key2',
    aggfunc='mean' # I often specify a function, even if it is the default, because I do not trust myself
)
practice_2


# In[47]:


np.allclose(practice_1, practice_2)


# Once you are comfortable with `pd.pivot_table()`, you could do the following:

# In[48]:


df.pivot_table(index='key1', columns='key2')


# We can specify a list of aggregation functions.

# In[49]:


df.pivot_table(index='key1', columns='key2', aggfunc=['mean', 'median', 'min', 'max'])


# ***Practice:***
# Calculate the sum of columns `a` through `e` by groups formed on the last letter in each name.
# *Hint:* use an anonymous (lambda) function.

# In[50]:


np.random.seed(42)
people = pd.DataFrame(
    data=np.random.randn(5, 5), 
    columns=['a', 'b', 'c', 'd', 'e'], 
    index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis']
)
people


# In[51]:


people.groupby(lambda x: x[-1]).sum()


# ***Practice:***
# Use the `.to_clipboard()` method to check your answer to the previous practice exercise.

# In[52]:


# people.to_clipboard() # only works on a local installation


# We need data for the following two practice exercises.
# We have to jump through some hoops with `pd.MultiIndex.from_product()` if we want to take full advantage of pandas multi indexes.

# In[53]:


faang = yf.download(tickers='META AAPL AMZN NFLX GOOG', session=session)
faang.columns.names = ['Variable', 'Ticker']
faang[pd.MultiIndex.from_product([['Return'], faang['Adj Close'].columns])] = faang['Adj Close'].pct_change()


# ***Practice:***
# For the FAANG stocks, calulate the mean and standard deviation of returns by ticker.

# In[54]:


faang.stack().groupby(level='Ticker').agg({'Return': ['mean', 'std']})


# ***Practice:***
# For the FAANG stocks, calulate the mean and standard deviation of returns and the maximum of closing prices by ticker.
# To do this, pass a dictionary where the keys are the column names and the values are lists of functions.

# In[55]:


faang.stack().groupby(level='Ticker').agg({'Return': ['mean', 'std'], 'Close': 'max'})


# What if we wanted these aggreations by ticker-month?
# (We will learn a slightly easier approach in chapter 11 of McKinney.)

# In[56]:


(
    faang # our original data with column multi-index for Variable and Ticker
    .stack() # moves Ticker from column inner level to row inner level, so data are Date-Ticker pairs
    .reset_index(level='Ticker') # removes Ticker from index, so index is Date only
    .groupby([
        'Ticker', # group by Ticker
        pd.Grouper(freq='M') # then, group by the month of each Date
    ])
    # .mean() # aggregates the mean of each column
    .agg({'Return': ['mean', 'std'], 'Close': 'max'}) # aggregates with column-specific functions
)


# We can check our work the old-fashioned way:

# In[57]:


faang.stack().loc[('1980-12', 'AAPL')].mean()


# In[58]:


faang.stack().loc[('2022-10', 'NFLX')].mean()


# ***Practice:***
# 
# 1. Download all available daily data for the S&P 500 ETF and Google stock (tickers SPY and GOOG)
# 2. Calculate daily returns
# 3. Calculate the volatility (standard deviation) of daily returns *every month* by combining `pd.Grouper()` and `.groupby()`)
# 4. Multiply by $\sqrt{252}$ to annualize these volatilities of daily returns
# 5. Plot these annualized volatilities

# In[59]:


ret_1 = yf.download(tickers='SPY GOOG', session=session)['Adj Close'].pct_change()
ret_1.columns.name = 'Ticker'


# In[60]:


vol_1 = ret_1.groupby(pd.Grouper(freq='M')).std()


# In[61]:


vol_1.dropna().mul(100 * np.sqrt(252)).plot()
plt.ylabel('Annualized Volatility (%)')
plt.title('Annualized Volatility from Daily Returns')
plt.show()


# In a few days, we will learn how to more easily perform this aggregation with the `.resample()` method.
# Still, `pd.Grouper()` is a great to tool to know because you may want to aggregate along several dimensions, which is not possible with the `.resample()` method.

# In[62]:


vol_2 = ret_1.resample('M').std() 


# In[63]:


np.allclose(vol_1, vol_2, equal_nan=True)


# ***Practice:***
# 
# 1. Download the daily factor data from Ken French's website
# 1. Calculate daily market returns by summing the market risk premium and risk-free rates (`Mkt-RF` and `RF`, respectively)
# 1. Calculate the volatility (standard deviation) of daily returns *every month* by combining `pd.Grouper()` and `.groupby()`)
# 1. Multiply by $\sqrt{252}$ to annualize these volatilities of daily returns
# 1. Plot these annualized volatilities
# 
# Is market volatility higher during wars?
# Consider the following dates:
# 
# 1. WWII: December 1941 to September 1945
# 1. Korean War: 1950 to 1953
# 1. Viet Nam War: 1959 to 1975
# 1. Gulf War: 1990 to 1991
# 1. War in Afghanistan: 2001 to 2021

# In[64]:


pdr.famafrench.get_available_datasets()[:5]


# In[65]:


ff = pdr.get_data_famafrench('F-F_Research_Data_Factors_daily', start='1900-01-01', session=session)[0] / 100


# In[66]:


ff['Mkt'] = ff['Mkt-RF'] + ff['RF']


# In[67]:


vol_3 = ff['Mkt'].groupby(pd.Grouper(freq='M')).std()


# In[68]:


vol_3.dropna().mul(100 * np.sqrt(252)).plot()
plt.axvspan('1941-12', '1945-09', alpha=0.25)
plt.annotate('WWII', ('1941-12', 90))
plt.axvspan('1950', '1953', alpha=0.25)
plt.annotate('Korean', ('1950', 80))
plt.axvspan('1959', '1975', alpha=0.25)
plt.annotate('Vietnam', ('1959', 90))
plt.axvspan('1990', '1991', alpha=0.25)
plt.annotate('Gulf I', ('1990', 80))
plt.axvspan('2001', '2021', alpha=0.25)
plt.annotate('Afghanistan', ('2001', 90))
plt.ylabel('Annualized Market Volatility (%)')
plt.title('Annualized Market Volatility from Daily Returns')
plt.show()

