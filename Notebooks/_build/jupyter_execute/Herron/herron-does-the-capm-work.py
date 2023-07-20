#!/usr/bin/env python
# coding: utf-8

# # Herron - Does the capital asset pricing model (CAPM) work?
# 
# This notebook uses the Ken French data library to show when the CAPM works and does not work.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

pd.set_option('display.float_format', '{:,.2f}'.format)
get_ipython().run_line_magic('precision', '4')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import pandas_datareader as pdr
import requests_cache
session = requests_cache.CachedSession(expire_after='1D')

import statsmodels.formula.api as smf


# Sadly, the empirical support for the CAPM is weak.
# The CAPM is:
# $$E[R_i] = R_f + \beta_i (E[R_{Market}] - R_f),$$
# which implies that higher $\beta$ stocks should have higher expected returns.
# We cannot know what investors expect, but their expectations should be similar to what they see on average.
# So we can look at the average annual returns for portfolios formed on CAPM $\beta$s from the previous year.
# Ken French provides these portfolios!

# In[2]:


[i for i in pdr.famafrench.get_available_datasets() if 'beta' in i.lower()]


# In[3]:


beta_all = pdr.get_data_famafrench('Portfolios_Formed_on_BETA', session=session, start='1900')


# In[4]:


print(beta_all['DESCR'])


# We will use the annual returns for value-weighted portfolios in `beta_all[2]`, although our results would be similar with different holding periods or different portfolio weights.

# In[5]:


beta_all[2].head()


# Ken French provides quintile portfolios (5 portfolios with 20% breakpoints) and decile portfolios (10 portfolios with 10% breakpoints).
# We will use the five quintile portfolios for simplicity, although our results would be similar with ten decile portfolios.

# In[6]:


beta_2 = beta_all[2].iloc[:, :5]


# We can also slices the average betas for each of these portfolios from `beta_all[6]` so we can plot the security market line (SML, the plot of returns versus $\beta$s).

# In[7]:


beta_6 = beta_all[6].iloc[:, :5]


# We will start with two bar charts:
# 
# 1. *Mean* annual returns for each beta portfolio
# 1. *Total* returns for each beta portfolio

# Plot 1 (mean annual returns for each $\beta$ portfolio) appears as we expect and shows a positive relation between returns and risk.
# The low $\beta$ portfolio has the lowest mean annual return and the high $\beta$ portfolio has the highest mean annual return.
# We find this positive and monotoic relation because the CAPM is a single-period model and works well for short holding periods.

# In[8]:


beta_2.mean().plot(kind='bar')
plt.xticks(rotation=0)
plt.ylabel('Mean Annual Return (%)')
plt.suptitle('Mean Annual Returns for Five Portfolios Formed on Beta')
plt.title(f'{beta_2.index[0]} to {beta_2.index[-1]}')
plt.show()


# However, plot 2 (*total* returns for each $\beta$ portfolio) does not support the CAPM.
# For example, the high $\beta$ portfolio has the lowest total return!
# The CAPM fails to predict longer horizon returns, which is our most common use case!

# In[9]:


beta_2.apply(lambda r: ((1 + r/100).prod() - 1)*100).plot(kind='bar')
plt.xticks(rotation=0)
plt.ylabel('Total Return (%)')
plt.suptitle('Total Returns for Five Portfolios Formed on Beta')
plt.title(f'{beta_2.index[0]} to {beta_2.index[-1]}')
plt.show()


# We gain another perspecitve if we plot the cumulative value of $1 invested in each of these portfolios at the beginning of the series (i.e., the beginning of 1963).

# In[10]:


beta_2.div(100).add(1).cumprod().plot()
plt.semilogy()
plt.ylabel('Value of $1 Investment ($)')
plt.suptitle('Value of $1 Investments in Five Portfolios Formed on Beta')
plt.title(f'{beta_2.index[0]} to {beta_2.index[-1]}')
plt.show()


# Finally, we can plot the SML by combining the annual portfolio returns in `beta_2` with their portfolio $\beta$s (from the previous year) in `beta_6`.

# In[11]:


df = (
    pd.concat(
        objs=[beta_2, beta_6], 
        keys=['Return', 'Beta'], 
        names=['Statistic', 'Portfolio'], 
        axis=1
    )
    .stack()
)


# The seaborn package makes it easy to create a scatter plot of returns versus $\beta$ with a best-fit line, which is the SML.

# In[12]:


df.dropna()


# In[13]:


df.index.get_level_values(1)


# In[14]:


import seaborn as sns
sns.regplot(x='Beta', y='Return', data=df)
plt.ylabel('Annual Return (%)')
plt.suptitle('Security Market Line (SML)')
plt.title(f'{df.dropna().index.get_level_values(0)[0]} to {df.dropna().index.get_level_values(0)[-1]}')
plt.show()


# The SML is nearly flat!
# Consitent with our second and third plots above, their is not a strong relation between returns and risk (as measured by $\beta$).
# We can estimate an ordinary least squares (OLS) regression to estimate the slope of the SML, which should be the market risk premium.

# In[15]:


import statsmodels.formula.api as smf
smf.ols(formula='Return ~ Beta', data=df).fit().summary()


# Again, we see that the slope of the SML is small (much less than the historical market risk premium of 5% to 7%).
# Further, the slope is not statistically different from zero!
# We can read more on the pros and cons of the CAPM in [chapter 10](https://book.ivo-welch.info/read/source5.mba/10-capm.pdf) (starting on page 15) of Ivo Welch's free corporate finance textbook.
