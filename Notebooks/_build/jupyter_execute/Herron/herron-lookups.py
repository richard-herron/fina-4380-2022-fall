#!/usr/bin/env python
# coding: utf-8

# # Herron - Lookups

# I will not know the answer to every question in class.
# When I do not, I will research and answer them here!

# ## 2022-09-14 Wednesday

# ### What is exactly is an `f` string?
# 
# [An `f` string is a new and improved way to format strings in Python](https://realpython.com/python-f-strings/#f-strings-a-new-and-improved-way-to-format-strings-in-python).
# Say I want to print "The office number for `professor` is `office`", where `professor` is a professor's name and `office` is her office number.
# The old way is to either concatenate (add) strings or use the `.format()` method.

# In[1]:


professor = 'Richard Herron'
office = 'Hayden Hall 120C'


# Concatenate strings:

# In[2]:


print('The office number for ' + professor + ' is ' + office)


# Use the `.format()` method:

# In[3]:


print('The office number for {} is {}'.format(professor, office))


# The `f` string format is easier to read:

# In[4]:


print(f'The office number for {professor} is {office}')


# Note we need the `f` in front parse the `{professor}` and `{office}`.
# [Here](https://peps.python.org/pep-0498/) is a more complete explanation.

# ## 2022-09-16 Friday

# ### Can we add attributes to a NumPy array?

# Yes!
# We can add attributes to a NumPy array: <https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray>.
# However, we will not add attributes to NumPy arrays or pandas data frames and series.
# Nor will we write our own classes because array, data frames, and series are feature complete for 99% of data analysis work.
# We would need to write our own classes if we wrote an econometrics package, but that is beyond this course and 99% of data analysis work.

# ### Are list comprehensions more than syntactic sugar?

# [Syntactic sugar](https://en.wikipedia.org/wiki/Syntactic_sugar) is:
# 
# > ...syntax within a programming language that is designed to make things easier to read or to express. It makes the language "sweeter" for human use: things can be expressed more clearly, more concisely, or in an alternative style that some may prefer. Syntactic sugar is usually a shorthand for a common operation that could also be expressed in an alternate, more verbose, form: The programmer has a choice of whether to use the shorter form or the longer form, but will usually use the shorter form since it is shorter and easier to type and read. 
# 
# I do not expect list comprehensions to be faster than for-loops, but it is hard to generalize and we would need to benchmark (`%%timeit`) specific cases.
# However, I consider the advantage of list comprehensions to be fast coding and comprehending instead of fast execution.
# [Here](https://realpython.com/list-comprehension-python/#benefits-of-using-list-comprehensions) is a clear discussion of the benefits of list comprehensions.

# ## 2022-09-23 Friday

# We know pandas, so I will add the pandas (and others) import statements and settings changes.

# In[5]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[6]:


plt.rcParams['figure.dpi'] = 150
get_ipython().run_line_magic('precision', '4')
pd.options.display.float_format = '{:.4f}'.format


# ### Can I update a data frame column with only new or changed values from a series?

# Here are some toy data.
# We will add the data in the series `ser` to the `b` and `c` columns in the data frame `df`.

# In[7]:


df = pd.DataFrame(np.arange(9).reshape(3, 3), index=['zero', 'one', 'two'], columns=list('abc'))
df.loc['one'] = np.nan


# In[8]:


df


# In[9]:


ser = pd.Series(100 * np.arange(3), index=['zero', 'one', 'two'])
ser.loc['zero'] = np.nan


# In[10]:


ser


# By default, pandas overwrites all column values with those from the series.
# Therefore, column `b` takes all the values of `ser`.

# In[11]:


df['b'] = ser


# In[12]:


df


# But can we keep the values in column `c` if the values in `ser` are missing?
# Yes, we can use the `.update()` method to update column `c` with the non-missing values in `ser`.

# In[13]:


df['c'].update(ser)


# In[14]:


df


# Series `ser` was missing the `zero` value, so `df` keeps its original 2.0000 value in the `zero` row and `c` column.
# However, `ser` was not missing the `one` and `two` values, so these values overwrite the `c` column values in `df`. 

# ### Can I ignore `NaN` values when I sum data frame rows, unless all values are `NaN`?

# Here is a data frame `df` that is missing one value in row `zero` and all values in row `one`.

# In[15]:


df = pd.DataFrame(np.arange(9).reshape(3, 3), index=['zero', 'one', 'two'], columns=list('abc'))
df.loc['zero', 'a'] = np.nan
df.loc['one'] = np.nan


# In[16]:


df


# The `.sum(axis=1)` method ignores `np.nan` by default.
# Therefore, the sum of the `zero` row is 3.000 and the sum of the `one` row is 0.0000, even though the `one` row has `np.nan` for every value.

# In[17]:


df.sum(axis=1)


# If we want a row sum of `np.nan` when every value is missing, we can use the `min_counts` argument in `.sum()`.
# If we require at least one value, the sum of row `one` becomes `np.nan`.

# In[18]:


df.sum(axis=1, min_count=1)


# If we require at least three non-missing values, the sum of row `zero` becomes `np.nan` because it only had two non-missing values.

# In[19]:


df.sum(axis=1, min_count=3)

