#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 4 - NumPy Basics: Arrays and Vectorized Computation

# ## Introduction
# 
# Chapter 4 of Wes McKinney's [*Python for Data Analysis*](https://wesmckinney.com/pages/book.html) discusses the NumPy package (short for numerical Python), which is the foundation for numerical computing in Python, including pandas.
# 
# We will focus on:
# 
# 1. Creating arrays
# 1. Slicing arrays
# 1. Performing mathematical operations on arrays
# 1. Applying functions and methods to arrays
# 1. Using conditional logic with arrays (i.e., `np.where()` and `np.select()`)
# 
# ***Note:*** Indented block quotes are from McKinney unless otherwise indicated. The section numbers here may differ from McKinney because we may not discuss every topic.

# The typical abbreviation for NumPy is `np`.

# In[1]:


import numpy as np


# The following prints NumPy arrays to 4 decimals places without changing the precision of the underlying array.

# In[2]:


get_ipython().run_line_magic('precision', '4')


# NumPy is critical to numerical computing in Python.
# We will often use NumPy via McKinney's pandas, but data analysts must know the fundamentals of NumPy.
# 
# > NumPy, short for Numerical Python, is one of the most important foundational packages for numerical computing in Python. Most computational packages providing scientific functionality use NumPy’s array objects as the lingua franca for data exchange.
# >
# > Here are some of the things you’ll find in NumPy:
# > - ndarray, an efficient multidimensional array providing fast array-oriented arithmetic operations and flexible broadcasting capabilities.
# > - Mathematical functions for fast operations on entire arrays of data without having to write loops.
# > - Tools for reading/writing array data to disk and working with memory-mapped files.
# > - Linear algebra, random number generation, and Fourier transform capabilities.
# > - A C API for connecting NumPy with libraries written in C, C++, or FORTRAN.
# >
# > Because NumPy provides an easy-to-use C API, it is straightforward to pass data to external libraries written in a low-level language and also for external libraries to return data to Python as NumPy arrays. This feature has made Python a language of choice for wrapping legacy C/C++/Fortran codebases and giving them a dynamic and
# easy-to-use interface. 
# >
# > While NumPy by itself does not provide modeling or scientific functionality, having an understanding of NumPy arrays and array-oriented computing will help you use tools with array-oriented semantics, like pandas, much more effectively. Since NumPy is a large topic, I will cover many advanced NumPy features like broadcasting
# in more depth later (see Appendix A). For most data analysis applications, the main areas of functionality I’ll focus on are: 
# > - Fast vectorized array operations for data munging and cleaning, subsetting and filtering, transformation, and any other kinds of computations
# > - Common array algorithms like sorting, unique, and set operations
# > - Efficient descriptive statistics and aggregating/summarizing data
# > - Data alignment and relational data manipulations for merging and joining together heterogeneous datasets
# > - Expressing conditional logic as array expressions instead of loops with if-elif-else branches
# > - Group-wise data manipulations (aggregation, transformation, function application)
# >
# > While NumPy provides a computational foundation for general numerical data processing, many readers will want to use pandas as the basis for most kinds of statistics or analytics, especially on tabular data. pandas also provides some more domain-specific functionality like time series manipulation, which is not present in NumPy.
# >
# > One of the reasons NumPy is so important for numerical computations in Python is because it is designed for efficiency on large arrays of data. There are a number of reasons for this:
# > - NumPy internally stores data in a contiguous block of memory, independent of other built-in Python objects. NumPy’s library of algorithms written in the C language can operate on this memory without any type checking or other overhead. NumPy arrays also use much less memory than built-in Python sequences.
# > - NumPy operations perform complex computations on entire arrays without the need for Python for loops.
# 
# 
# McKinney provides a clear example of NumPy's power and speed.
# He creates 2 sequences of numbers from 0 to 999,999 as a NumPy array and a Python list, then multiplies the list by 2.
# The NumPy array supports vectorized operations and it "just works" when he multiplies the array by 2.
# However, he must use a list comprehension to multiply the list by 2 element-by-element.

# In[3]:


my_list = list(range(1000000))


# In[4]:


my_arr = np.arange(1000000)


# In[5]:


my_list[:5]


# In[6]:


my_arr[:5]


# Multiplying lists by an integer concatenates them instead of elementwise multiplication.
# So we use a list comprehension to multiply the elements in `my_list` by 2.

# In[7]:


# my_list * 2 # concatenates two copies of my_list


# In[8]:


# [2 * x for x in my_list] # we use a list comprehension for elementwise multiplication of a list


# However, math on NumPy arrays "just works".

# In[9]:


my_arr * 2


# We use the "magic" function `%timeit` to time these two calculations.

# In[10]:


get_ipython().run_line_magic('timeit', '[x * 2 for x in my_list]')


# In[11]:


get_ipython().run_line_magic('timeit', 'my_arr * 2')


# The NumPy version is few hundred times faster than the list version.

# ## The NumPy ndarray: A Multidimensional Array Object
# 
# > One of the key features of NumPy is its N-dimensional array object, or ndarray, which is a fast, flexible container for large datasets in Python. Arrays enable you to perform mathematical operations on whole blocks of data using similar syntax to the equivalent operations between scalar elements.
# 
# We can generate random data to explore NumPy arrays.
# Whenever we use random data, we should set the random number seed with `np.random.seed(42)`, which makes our random numbers repeatable.
# Because we set the random number seed just before we generate `data`, our `data`s will be identical.

# In[12]:


import this


# In[13]:


np.random.seed(42)
data = np.random.randn(2, 3)


# In[14]:


data


# Multiplying `data` by 10 multiplies each element in `data` by 10, and adding `data` to itself does element-wise addition.
# The compromise to achieve this common-sense behavior is that NumPy arrays must contain homogeneous data types (e.g., all floats or all integers).

# In[15]:


data * 10


# Addition in NumPy is also elementwise.

# In[16]:


data_2 = data + data


# NumPy arrays also have attributes.
# Recall that Jupyter Notebooks provides tab completion.

# In[17]:


data.shape


# In[18]:


data.dtype


# In[19]:


data[0]


# In[20]:


data[0][0] # zero row, then the zero element in the zero row


# In[21]:


data[0, 0] # zero row, zero column


# ### Creating ndarrays
# 
# > The easiest way to create an array is to use the array function. This accepts any sequence-like object (including other arrays) and produces a new NumPy array containing the passed data

# In[22]:


data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)


# In[23]:


arr1


# In[24]:


arr1.dtype


# We could coerce these values to integers, but we would lose information.
# The default is to select a `dtype` that would not lose information.

# In[25]:


np.array(data1, dtype=np.int64)


# Note that `np.array()` re-cast the values in `data1` to floats becuase NumPy arrays must be homogenous data types.
# A list of lists becomes a two-dimensional array.

# In[26]:


data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)


# In[27]:


arr2


# In[28]:


arr2.ndim


# In[29]:


arr2.shape


# In[30]:


arr2.dtype


# There are several other ways to create NumPy arrays.

# In[31]:


np.zeros(10)


# In[32]:


np.zeros((3, 6))


# The `np.arange()` function is similar to the core `range()` but it creates an array directly.

# In[33]:


list(range(15))


# In[34]:


np.arange(15)


# ***Table 4-1*** lists the NumPy array creation functions.
# 
# - `array`: Convert input data (list, tuple, array, or other sequence type) to an ndarray either by inferring a dtype or explicitly specifying a dtype; copies the input data by default
# - `asarray`:  Convert input to ndarray, but do not copy if the input is already an ndarray 
# - `arange`:  Like the built-in range but returns an ndarray instead of a list
# - `ones`, `ones_like`:  Produce an array of all 1s with the given shape and dtype; `ones_like` takes another array and produces a `ones` array of the - same shape and dtype
# - `zeros`, `zeros_like`:  Like `ones` and `ones_like` but producing arrays of 0s instead
# - `empty`, `empty_like`:  Create new arrays by allocating new memory, but do not populate with any values like ones and zeros
# - `full`, `full_like`:  Produce an array of the given shape and dtype with all values set to the indicated "fill value"
# - `eye`, `identity`:  Create a square N-by-N identity matrix (1s on the diagonal and 0s elsewhere)

# ### Arithmetic with NumPy Arrays
# 
# > Arrays are important because they enable you to express batch operations on data without writing any for loops. NumPy users call this vectorization. Any arithmetic operations between equal-size arrays applies the operation element-wise

# In[35]:


arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr


# NumPy array addition is elementwise.

# In[36]:


arr + arr


# NumPy array multiplication is elementwise.

# In[37]:


arr * arr


# NumPy array division is elementwise.

# In[38]:


1 / arr


# In[39]:


arr ** 0.5


# In[40]:


arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])


# ### Basic Indexing and Slicing
# 
# One-dimensional array index and slice the same as lists.

# In[41]:


arr = np.arange(10)
arr


# In[42]:


arr[5]


# In[43]:


arr[5:8]


# In[44]:


equiv_list = list(range(10))
equiv_list


# In[45]:


equiv_list[5:8]


# In[46]:


# equiv_list[5:8] = 12 # TypeError: can only assign an iterable


# In[47]:


equiv_list[5:8] = [12] * 3
equiv_list


# With NumPy arrays, we do not have to ump through this hoop.

# In[48]:


arr[5:8] = 12
arr


# > As you can see, if you assign a scalar value to a slice, as in `arr[5:8] = 12`, the value is propagated (or broadcasted henceforth) to the entire selection. An important first distinction from Python’s built-in lists is that array slices are views on the original array. This means that the data is not copied, and any modifications to the view will be reflected in the source array.

# In[49]:


arr_slice = arr[5:8]
arr_slice


# In[50]:


x = arr_slice


# In[51]:


x


# In[52]:


x is arr_slice


# In[53]:


y = x.copy()


# In[54]:


y is x


# In[55]:


arr_slice[1] = 12345
arr_slice


# In[56]:


arr


# The `:` slices every element in `arr_slice`.

# In[57]:


arr_slice[:] = 64
arr_slice


# In[58]:


arr


# > If you want a copy of a slice of an ndarray instead of a view, you will need to explicitly copy the array-for example, `arr[5:8].copy()`.

# In[59]:


arr_slice_2 = arr[5:8].copy()
arr_slice_2


# In[60]:


arr_slice_2[:] = 2001


# In[61]:


arr_slice_2


# In[62]:


arr


# > With higher dimensional arrays, you have many more options. In a two-dimensional array, the elements at each index are no longer scalars but rather one-dimensional arrays... Thus, individual elements can be accessed recursively. But that is a bit too much work, so you can pass a comma-separated list of indices to select individual elements.

# In[63]:


arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d


# In[64]:


arr2d[2]


# In[65]:


arr2d[0][2]


# In[66]:


arr2d[0, 2] # row, column notation is more common and typically easier to read


# ### Indexing with slices

# In[67]:


arr2d


# In[68]:


arr2d[:2]


# In[69]:


arr2d[:2, 1:]


# In[70]:


arr2d[1, :2]


# In[71]:


arr2d[:2, 2]


# A colon (`:`) by itself selects the entire dimension and is necessary to slice higher dimensions.

# In[72]:


arr2d[:, :1]


# In[73]:


arr2d[:2, 1:] = 0
arr2d


# ***ALWAYS CHECK YOUR OUTPUT!***

# ### Boolean Indexing
# 
# We can use Booleans (`True`s and `False`s) to slice, too.
# Think of `names` as a sequence of seven names that line up with the seven rows in `data`.
# To keep things simple, we will not give column names.
# The folowing example is like `index(match(), match())` from Excel.

# In[74]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.random.seed(42)
data = np.random.randn(7, 4)


# In[75]:


names


# In[76]:


data


# In[77]:


names == 'Bob'


# In[78]:


data[names == 'Bob']


# In[79]:


data[names == 'Bob', 2:]


# In[80]:


data[names == 'Bob', 3]


# In[81]:


names != 'Bob'


# The `~` inverts a Boolean condition.

# In[82]:


data[~(names == 'Bob')]


# In[83]:


cond = names == 'Bob'
data[~cond]


# For NumPy arrays, we must use `&` and `|` instead of `and` and `or`.

# In[84]:


mask = (names == 'Bob') | (names == 'Will')


# In[85]:


data[mask]


# The "not" operator is `~` in NumPy.
# The "not" operator is typically `!` in other programming languages.

# In[86]:


data[~(names == 'Bob')]


# In[87]:


data


# In[88]:


data < 0


# In[89]:


data[data < 0] = 0


# In[90]:


data


# In[91]:


data[names != 'Joe'] = 7


# In[92]:


data


# Note:
# 
# > Selecting data from an array by boolean indexing always creates a copy of the data,
# even if the returned array is unchanged.

# ## Universal Functions: Fast Element-Wise Array Functions
# 
# > A universal function, or ufunc, is a function that performs element-wise operations on data in ndarrays. You can think of them as fast vectorized wrappers for simple functions that take one or more scalar values and produce one or more scalar results.

# In[93]:


arr = np.arange(10)


# In[94]:


np.sqrt(arr)


# Here `np.exp(x)` is $e^x$.

# In[95]:


np.exp(arr)


# In[96]:


2**arr


# > These are referred to as unary ufuncs. Others, such as add or maximum, take two arrays (thus, binary ufuncs) and return a single array as the result.

# In[97]:


np.random.seed(42)
x = np.random.randn(8)
y = np.random.randn(8)


# In[98]:


np.maximum(x, y)


# ***Be careful! Function names are not the whole story. Check your output and read the docstring!***

# In[99]:


np.max(x)


# ***Table 4-3*** lists the fast, element-wise unary functions:
# 
# - `abs`, `fabs`: Compute the absolute value element-wise for integer, oating-point, or complex values
# - `sqrt`: Compute the square root of each element (equivalent to arr ** 0.5) 
# - `square`: Compute the square of each element (equivalent to arr ** 2)
# - `exp`: Compute the exponent $e^x$ of each element
# - `log`, `log10`, `log2`, `log1p`: Natural logarithm (base e), log base 10, log base 2, and log(1 + x), respectively
# - `sign`: Compute the sign of each element: 1 (positive), 0 (zero), or –1 (negative)
# - `ceil`: Compute the ceiling of each element (i.e., the smallest integer greater than or equal to thatnumber)
# - `floor`: Compute the oor of each element (i.e., the largest integer less than or equal to each element)
# - `rint`: Round elements to the nearest integer, preserving the dtype
# - `modf`: Return fractional and integral parts of array as a separate array
# - `isnan`: Return boolean array indicating whether each value is NaN (Not a Number)
# - `isfinite`, `isinf`: Return boolean array indicating whether each element is finite (non-inf, non-NaN) or infinite, respectively
# - `cos`, `cosh`, `sin`, `sinh`, `tan`, `tanh`: Regular and hyperbolic trigonometric functions
# - `arccos`, `arccosh`, `arcsin`, `arcsinh`, `arctan`, `arctanh`: Inverse trigonometric functions
# - `logical_not`: Compute truth value of not x element-wise (equivalent to ~arr).

# ***Table 4-4*** lists the fast, element-wise binary functions:
# 
# - `add`: Add corresponding elements in arrays
# - `subtract`: Subtract elements in second array from first array
# - `multiply`: Multiply array elements
# - `divide`, `floor_divide`: Divide or floor divide (truncating the remainder)
# - `power`: Raise elements in first array to powers indicated in second array
# - `maximum`, `fmax`: Element-wise maximum; `fmax` ignores `NaN`
# - `minimum`, `fmin`: Element-wise minimum; `fmin` ignores `NaN`
# - `mod`: Element-wise modulus (remainder of division)
# - `copysign`: Copy sign of values in second argument to values in first argument
# - `greater`, `greater_equal`, `less`, `less_equal`, `equal`, `not_equal`: Perform element-wise comparison, yielding boolean array (equivalent to infix operators >, >=, <, <=, ==, !=)
# - `logical_and`, `logical_or`, `logical_xor`: Compute element-wise truth value of logical operation (equivalent to infix operators & |, ^)

# ## Array-Oriented Programming with Arrays
# 
# > Using NumPy arrays enables you to express many kinds of data processing tasks as concise array expressions that might otherwise require writing loops. This practice of replacing explicit loops with array expressions is commonly referred to as vectorization. In general, vectorized array operations will often be one or two (or more) orders of magnitude faster than their pure Python equivalents, with the biggest impact in any kind of numerical computations. Later, in Appendix A, I explain broadcasting, a powerful method for vectorizing computations.

# ### Expressing Conditional Logic as Array Operations
# 
# > The numpy.where function is a vectorized version of the ternary expression x if condition else y.

# In[100]:


xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])


# In[101]:


result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]


# In[102]:


result


# NumPy's `where()` is an if-else statement that operates like Excel's `if()`.

# In[103]:


np.where(cond, xarr, yarr)


# We could also use `np.select()`.

# In[104]:


np.select(
    condlist=[cond==True, cond==False],
    choicelist=[xarr, yarr]
)


# ### Mathematical and Statistical Methods
# 
# > A set of mathematical functions that compute statistics about an entire array or about the data along an axis are accessible as methods of the array class. You can use aggregations (often called reductions) like sum, mean, and std (standard deviation) either by calling the array instance method or using the top-level NumPy function.
# 
# We will use these aggregations extensively in pandas.

# In[105]:


np.random.seed(42)
arr = np.random.randn(5, 4)


# In[106]:


arr.mean()


# In[107]:


np.mean(arr)


# In[108]:


arr.sum()


# The aggregation methods above aggregated the whole array.
# We can use the `axis` argument to aggregate columns (`axis=0`) and rows (`axis=1`).

# In[109]:


arr.mean(axis=1)


# In[110]:


arr[0].mean()


# In[111]:


arr.mean(axis=0)


# In[112]:


arr = np.array([0, 1, 2, 3, 4, 5, 6, 7])
arr.cumsum()


# In[113]:


arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])


# In[114]:


arr


# In[115]:


arr.cumsum(axis=0)


# In[116]:


arr.cumprod(axis=1)


# Table 4-5 lists the basic statistical methods:
# 
# - `sum`: Sum of all the elements in the array or along an axis; zero-length arrays have sum 0
# - `mean`: Arithmetic mean; zero-length arrays have NaN mean
# - `std`, `var`: Standard deviation and variance, respectively, with optional degrees of freedom adjustment (default denominator $n$)
# - `min`, `max`: Minimum and maximum
# - `argmin`, `argmax`: Indices of minimum and maximum elements, respectively
# - `cumsum`: Cumulative sum of elements starting from 0
# - `cumprod`: Cumulative product of elements starting from 1

# NumPy's `.var()` and `.std()` methods return *population* statistics (denominators of $n$).
# The pandas equivalents return *sample* statistics (denominators of $n-1$), which are more appropriate for financial data analysis where we have a sample instead of a population.

# ### Methods for Boolean Arrays

# In[117]:


np.random.seed(42)
arr = np.random.randn(100)


# In[118]:


(arr > 0).sum() # Number of positive values


# In[119]:


(arr > 0).mean() # percentage of positive values


# In[120]:


bools = np.array([False, False, True, False])


# In[121]:


bools.any()


# In[122]:


bools.all()


# ### Sorting

# > Like Python's built-in list type, NumPy arrays can be sorted in-place with the sort method.

# In[123]:


np.random.seed(42)
arr = np.random.randn(6)


# In[124]:


arr.sort()


# In[125]:


np.random.seed(42)
arr = np.random.randn(5, 3)


# For two-dimensional arrays (and beyond), we can sort along an axis.
# The default sort axis is the last axis.
# Recall that `axis=1` operates on rows.
# Note that these row sorts are independent of one another.

# In[126]:


arr.sort(1)


# ## Practice

# ***Practice:***
# Create a 1-dimensional array named `a1` that counts from 0 to 24 by 1.

# In[127]:


a1 = np.arange(25)
a1


# ***Practice:***
# Create a 1-dimentional array named `a2` that counts from 0 to 24 by 3.

# In[128]:


a2 = np.arange(0, 25, 3)
a2


# ***Practice:***
# Create a 1-dimentional array named `a3` that counts from 0 to 100 by multiples of 3 and 5.

# In[129]:


np.array([i for i in range(101) if i%3==0 or i%5==0])


# ***Practice:***
# Create a 1-dimensional array `a3` that contains the squares of the even integers through 100,000.
# How much faster is the NumPy version than the list comprehension version?

# In[130]:


a3 = np.arange(0, 100001, 2)**2
a3


# In[131]:


get_ipython().run_line_magic('timeit', 'np.arange(0, 100001, 2)**2')


# In[132]:


get_ipython().run_line_magic('timeit', '[i**2 for i in range(0, 100001, 2)]')


# ***Practice:***
# Write functions that mimic Excel's `pv` and `fv` functions.

# In[133]:


def pv(rate, nper, pmt, fv=0, end=True):
    _pv_pmt = (pmt / rate) * (1 - (1 + rate)**(-nper))
    if not end: _pm_pmt *= (1 + rate)
    _pv_fv = fv * (1 + rate)**(-nper)
    return -1 * (_pv_pmt + _pv_fv)


# In[134]:


pv(0.1, 10, 10, 100)


# In[135]:


def fv(rate, nper, pmt, fv=0, end=True):
    _fv_pmt = (pmt / rate) * ((1 + rate)**nper - 1)
    if not end: _pm_pmt *= (1 + rate)
    _fv_fv = fv * (1 + rate)**nper
    return -1 * (_fv_pmt + _fv_fv)


# In[136]:


fv(0.1, 10, 10, 100)


# ***Practice***
# Create a copy of `data` named `data2`, and replace negative values with -1 and positive values with +1.

# In[137]:


np.random.seed(42)
data = np.random.randn(7, 4)


# In[138]:


data2 = data.copy()
data2[data2 < 0] = -1
data2[data2 > 0] = +1
data2


# ***Or***, we can use `np.select()`.

# In[139]:


np.select(
    condlist=[data<0, data>0],
    choicelist=[-1, +1],
    default=data
)


# ***Practice:***
# Write a function that calculates the number of payments that generate $x\%$ of the present value of a perpetuity given $C_1$, $r$, and $g$.
# Recall the present value of a growing perpetuity is $PV = \frac{C_1}{r - g}$ and the present value of a growing annuity is $PV = \frac{C_1}{r - g}\left[ 1 - \left( \frac{1 + g}{1 + r} \right)^t \right]$.

# In[140]:


def npmt(x, r, g):
    return np.log(1 - x) / np.log((1 + g) / (1 + r))


# In[141]:


npmt(0.5, 0.1, 0.05)


# We do not need to use $C_1$ because it cancels from both sides of the $=$.
# So
# $$x \times \frac{C_1}{r-g} = \frac{C_1}{r-g} \left[ 1 - \left( \frac{1 + g}{1 + r} \right)^t \right],$$
# becomes
# $$x = \left[ 1 - \left( \frac{1 + g}{1 + r} \right)^t \right],$$
# which we can manipulate to solve for $t$:
# $$t = \log(1 - x) \div \log\left(\frac{1+g}{1+r}\right).$$
# We can verify this with present value of perpetuity and annuity functions.

# In[142]:


def pv_perp(c, r, g):
    return c / (r - g)


# In[143]:


def pv_ann(c, r, g, t):
    return pv_perp(c, r, g) * (1 - ((1 + g) / (1 + r))**t)


# In[144]:


pv_ann(1, 0.1, 0.05, 14.899977377480532) / pv_perp(1, 0.1, 0.05)


# ***Practice:***
# Write a function that calculates the internal rate of return given an numpy array of cash flows.

# In[145]:


cf = np.array([-100, 50, 50, 50])


# In[146]:


def npv(r, cf):
    _t = np.arange(cf.shape[0])
    return (cf / (1 + r)**_t).sum()


# In[147]:


def irr(cf, guess=0, tol=1e-6, step=1e-6):
    
    _r = guess # our first guess, recall we can have multiple IRRs, so we may need to guess
    _npv = npv(_r, cf) # IRR is the rate where NPV equals 0
    while np.abs(_npv) > tol: # while the absolute value of NPV does not equal 0...
        _r += _npv * step # increase the discount if NPV < 0, otherwise decrease the discount rate
        _npv = npv(_r, cf) # re-calculate NPV with new discount rate
    
    return _r


# In[148]:


irr(cf)


# We can check that NPV is zero at this rate:

# In[149]:


npv(irr(cf), cf)


# Excel and your financial calculator use a more sophisticated algorithm.
# We can revisit this problem once we know more Python.
