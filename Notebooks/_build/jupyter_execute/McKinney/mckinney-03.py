#!/usr/bin/env python
# coding: utf-8

# # McKinney Chapter 3 - Built-in Data Structures, Functions, and Files

# ## Introduction
# 
# We must understand Python's core functionality to make full use of NumPy and pandas.
# Chapter 3 of Wes McKinney's [*Python for Data Analysis*](https://wesmckinney.com/pages/book.html) discusses Python's core functionality.
# We will focus on the following:
# 
# 1. Data structures (we will ignore sets)
#     1. tuples
#     1. lists
#     1. dicts (also known as dictionaries)
# 1. List comprehensions (we will ignore set and dict comprehensions)
# 1. Functions
#     1. Returning multiple values
#     1. Using anonymous functions
# 
# ***Note:*** Block quotes are from McKinney, and section numbers differ from McKinney because we will not discuss every topic.

# ## Data Structures and Sequences
# 
# > Python's data structures are simple but powerful. Mastering their use is a critical part
# of becoming a proficient Python programmer.

# ### Tuple
# 
# > A tuple is a fixed-length, immutable sequence of Python objects.
# 
# We cannot change a tuple after we create it, so tuples are immutable.
# A tuple is ordered, so we can subset or slice it with a numerical index.
# We will surround tuples with parentheses but the parentheses are not always required.

# In[1]:


tup = (4, 5, 6) # the parentheses are optional, but we will use them


# In[2]:


# tup = 4, 5, 6 # this is the same as the previous code cell because the parentheses are optional


# In[3]:


tup[0] # PYTHON IS ZERO-INDEXED!


# In[4]:


tup[1]


# In[5]:


nested_tup = ((4, 5, 6), (7, 8))


# In[6]:


nested_tup[0] # PYTHON IS ZERO-INDEXED!


# In[7]:


nested_tup[0][0]


# In[8]:


tuple([4, 0, 2])


# In[9]:


tup = tuple('string')


# In[10]:


tup


# In[11]:


tup[0]


# In[12]:


tup = tuple(['foo', [1, 2], True])


# In[13]:


tup


# In[14]:


# tup[2] = False # gives an error, because tuples are immutable (unchangeable)


# > If an object inside a tuple is mutable, such as a list, you can modify it in-place.

# In[15]:


tup


# > You can concatenate tuples using the + operator to produce longer tuples:
# 
# Tuples are immutable.
# So we cannot modify a tuple, but we can combine two tuples into a new tuple.

# In[16]:


(1, 2) + (1, 2)


# In[17]:


(4, None, 'foo') + (6, 0) + ('bar',)


# > Multiplying a tuple by an integer, as with lists, has the effect of concatenating together
# that many copies of the tuple:
# 
# This multiplication behavior is the logical extension of the addition behavior above.
# The output of `tup + tup` should be the same as the output of `2 * tup`.

# In[18]:


('foo', 'bar') * 4


# In[19]:


('foo', 'bar') + ('foo', 'bar') + ('foo', 'bar') + ('foo', 'bar')


# In[20]:


(1, 2) * 4


# #### Unpacking tuples
# 
# > If you try to assign to a tuple-like expression of variables, Python will attempt to
# unpack the value on the righthand side of the equals sign.

# In[21]:


tup = (4, 5, 6)
a, b, c = tup


# In[22]:


a


# In[23]:


b


# In[24]:


c


# In[25]:


(d, e, f) = (7, 8, 9) # the parentheses are optional but helpful


# In[26]:


d


# In[27]:


e


# In[28]:


f


# In[29]:


# g, h = 10, 11, 12 # ValueError: too many values to unpack (expected 2)


# We can even unpack nested tuples!

# In[30]:


tup = 4, 5, (6, 7)
a, b, (c, d) = tup


# In[31]:


a


# In[32]:


b


# In[33]:


c


# In[34]:


d


# You can use this functionality to rename variables without a using a third, temporary variable.

# #### Tuple methods
# 
# > Since the size and contents of a tuple cannot be modified, it is very light on instance
# methods. A particularly useful one (also available on lists) is count, which counts the
# number of occurrences of a value.

# In[35]:


a = (1, 2, 2, 2, 3, 4, 2)
a.count(2)


# In[36]:


import this


# ### List
# 
# > In contrast with tuples, lists are variable-length and their contents can be modified in-place. You can define them using square brackets [ ] or using the list type function.

# In[37]:


a_list = [2, 3, 7, None]
tup = ('foo', 'bar', 'baz')
b_list = list(tup)


# In[38]:


a_list


# In[39]:


b_list


# In[40]:


a_list[0] # PYTHON IS ZERO-INDEXED!


# #### Adding and removing elements
# 
# > Elements can be appended to the end of the list with the append method.
# 
# The `.append()` method appends an element to the list *in place* without reassigning the list.

# In[41]:


b_list.append('dwarf')


# In[42]:


b_list


# > Using insert you can insert an element at a specific location in the list.
# The insertion index must be between 0 and the length of the list, inclusive.

# In[43]:


b_list.insert(1, 'red') # PYTHON IS ZERO INDEXED


# In[44]:


b_list


# In[45]:


b_list.index('red')


# In[46]:


b_list[b_list.index('red')] = 'blue'


# In[47]:


b_list


# > The inverse operation to insert is pop, which removes and returns an element at a
# particular index.

# In[48]:


b_list.pop(2)


# In[49]:


b_list


# Note that `.pop(2)` removes the 2 element.
# If we do not want to remove the 2 element, we should use `[2]` to access an element without removing it.

# > Elements can be removed by value with remove, which locates the first such value and removes it from the list.

# In[50]:


b_list.append('foo')


# In[51]:


b_list


# In[52]:


b_list.remove('foo')


# In[53]:


b_list


# > If performance is not a concern, by using append and remove, you can use a Python list as a perfectly suitable "multiset" data structure.
# 
# However, appending to and removing from a list is very slow.
# When an example arises, we will benchmark appending to a list against alternatives.

# In[54]:


'dwarf' in b_list


# In[55]:


'dwarf' not in b_list


# #### Concatenating and combining lists
# 
# > Similar to tuples, adding two lists together with + concatenates them.

# In[56]:


[4, None, 'foo'] + [7, 8, (2, 3)]


# In[57]:


xx = [4, None, 'foo']
xx.append([7, 8, (2, 3)])


# In[58]:


xx


# > If you have a list already defined, you can append multiple elements to it using the extend method.

# In[59]:


x = [4, None, 'foo']
x.extend([7, 8, (2, 3)])


# In[60]:


x


# ***CHECK YOUR OUTPUT! IT IS DIFFICULT TO MEMORIZE ALL THESE METHODS!!!***

# In[61]:


xxx = [4, None, 'foo']
xxx.append((7, 8, (2, 3)))


# In[62]:


xxx


# In[63]:


type([1,2,3])


# In[64]:


type((1,2,3))


# #### Sorting
# 
# > You can sort a list in-place (without creating a new object) by calling its sort function.

# In[65]:


a = [7, 2, 5, 1, 3]
a.sort()


# In[66]:


a


# > sort has a few options that will occasionally come in handy. One is the ability to pass a secondary sort key—that is, a function that produces a value to use to sort the objects. For example, we could sort a collection of strings by their lengths.
# 
# Before you write your own solution to a problem, read the docstring (help file) of the built-in function.
# The built-in function may already solve your problem (typically faster and with fewer bugs).

# In[67]:


b = ['saw', 'small', 'He', 'foxes', 'six']
b.sort()


# In[68]:


b # Python is case sensitive, so "He" sorts before "foxes"


# In[69]:


b.sort(key=len)


# In[70]:


b


# #### Slicing
# 
# ***Slicing is very important!***
# 
# > You can select sections of most sequence types by using slice notation, which in its basic form consists of start:stop passed to the indexing operator [ ].
# 
# Recall that Python is zero-indexed, so the first element has an index of 0.
# The necessary consequence of zero-indexing is that start:stop is inclusive on the left edge (start) and exclusive on the right edge (stop).

# In[71]:


seq = [7, 2, 3, 7, 5, 6, 0, 1]


# In[72]:


seq[5]


# In[73]:


seq[:5] # we read the ":5" slice as "0:5"


# In[74]:


seq[1:5]


# In[75]:


seq[3:5]


# > Either the start or stop can be omitted, in which case they default to the start of the sequence and the end of the sequence, respectively.

# In[76]:


seq[:5]


# In[77]:


seq[3:]


# > Negative indices slice the sequence relative to the end.

# In[78]:


seq


# In[79]:


seq[-1:]


# In[80]:


seq[-4:]


# In[81]:


seq[-4:-1]


# In[82]:


seq[-6:-2]


# > A step can also be used after a second colon to, say, take every other element.

# In[83]:


seq


# In[84]:


seq[::2]


# In[85]:


seq[1::2]


# I remember the trick above as `:2` is "count by 2".

# > A clever use of this is to pass -1, which has the useful effect of reversing a list or tuple.

# In[86]:


seq[::-1]


# We will use slicing (subsetting) all semester, so it is worth a few minutes to understand the examples above.

# ### dict
# 
# > dict is likely the most important built-in Python data structure. A more common
# name for it is hash map or associative array. It is a flexibly sized collection of key-value
# pairs, where key and value are Python objects. One approach for creating one is to use
# curly braces {} and colons to separate keys and values.
# 
# Elements in dictionaries have names, while elements in tuples and lists have numerical indices.
# Dictionaries are handy for passing named arguments and returning named results.

# In[87]:


empty_dict = {}


# In[88]:


empty_dict


# A dictionary is a set of key-value pairs.

# In[89]:


d1 = {'a' : 'some value', 'b' : [1, 2, 3, 4]}


# In[90]:


d1['a']


# In[91]:


d1[7] = 'an integer'


# In[92]:


d1


# We access dictionary elements by name/label instead of position.

# In[93]:


d1['b']


# In[94]:


'b' in d1


# > You can delete values either using the del keyword or the pop method (which simultaneously returns the value and deletes the key).

# In[95]:


d1[5] = 'some value'


# In[96]:


d1['dummy'] = 'another value'


# In[97]:


d1


# In[98]:


del d1[5]


# In[99]:


d1


# In[100]:


ret = d1.pop('dummy')


# In[101]:


ret


# In[102]:


d1


# > The keys and values method give you iterators of the dict’s keys and values, respectively. While the key-value pairs are not in any particular order, these functions output the keys and values in the same order.

# In[103]:


d1.keys()


# In[104]:


d1.values()


# > You can merge one dict into another using the update method.

# In[105]:


d1.update({'b' : 'foo', 'c' : 12})


# In[106]:


d1


# We will skip the rest of the dictionary topics from McKinney, as we are unlikely to use them.

# ## List, Set, and Dict Comprehensions
# 
# > List comprehensions are one of the most-loved Python language features. They allow you to concisely form a new list by filtering the elements of a collection, transforming the elements passing the filter in one concise expression. They take the basic form:
# > ```python
# > [expr for val in collection if condition]
# > ```
# > This is equivalent to the following for loop:
# > ```python
# > result = []
# > for val in collection:
# >     if condition:
# >     result.append(expr)
# > ```
# > The filter condition can be omitted, leaving only the expression.
# 
# We will focus on list comprehensions, which are very [Pythonic](https://blog.startifact.com/posts/older/what-is-pythonic.html).

# In[107]:


strings = ['a', 'as', 'bat', 'car', 'dove', 'python']


# In[108]:


caps = []
for x in strings:
    if len(x) > 2:
        caps.append(x.upper())
caps


# We can replace the for loop above with a list comprehension.

# In[109]:


# [operation on x for x in list if condition is met]
[x.upper() for x in strings if len(x) > 2]


# Here is another example.
# Write a for-loop and the equivalent list comprehension that squares the integers from 1 to 10.

# In[110]:


squares = []
for i in range(1, 11):
    squares.append(i ** 2)
    
squares


# In[111]:


squares_2 = [i**2 for i in range(1, 11)]

squares_2


# ## Functions
# 
# > Functions are the primary and most important method of code organization and reuse in Python. As a rule of thumb, if you anticipate needing to repeat the same or very similar code more than once, it may be worth writing a reusable function. Functions can also help make your code more readable by giving a name to a group of Python statements.
# >
# > Functions are declared with the def keyword and returned from with the return keyword:
# > ```python
# > def my_function(x, y, z=1.5):
# >     if z > 1:
# >          return z * (x + y)
# >      else:
# >          return z / (x + y)
# > ```
# > There is no issue with having multiple return statements. If Python reaches the end of a function without encountering a return statement, None is returned automatically.
# >
# > Each function can have positional arguments and keyword arguments. Keyword arguments are most commonly used to specify default values or optional arguments. In the preceding function, x and y are positional arguments while z is a keyword argument. This means that the function can be called in any of these ways:
# > ```python
# >  my_function(5, 6, z=0.7)
# >  my_function(3.14, 7, 3.5)
# >  my_function(10, 20)
# > ```
# > The main restriction on function arguments is that the keyword arguments must follow the positional arguments (if any). You can specify keyword arguments in any order; this frees you from having to remember which order the function arguments were specified in and only what their names are.
# 
# Here is the basic syntax for a function:

# In[112]:


def mult_by_two(x):
    return 2*x


# ### Returning Multiple Values
# 
# We can write Python functions that return multiple objects.
# In reality, the function `f()` below returns one object, a tuple, that we can unpack to multiple objects.

# In[113]:


def f():
    a = 5
    b = 6
    c = 7
    return a, b, c


# If we want to return multiple objects with names or labels, we can return a dictionary.

# In[114]:


def f():
    a = 5
    b = 6
    c = 7
    return {'a' : a, 'b' : b, 'c' : c}


# In[115]:


f()['a']


# ### Anonymous (Lambda) Functions
# 
# > Python has support for so-called anonymous or lambda functions, which are a way of writing functions consisting of a single statement, the result of which is the return value. They are defined with the lambda keyword, which has no meaning other than "we are declaring an anonymous function."
# 
# > I usually refer to these as lambda functions in the rest of the book. They are especially convenient in data analysis because, as you'll see, there are many cases where data transformation functions will take functions as arguments. It's often less typing (and clearer) to pass a lambda function as opposed to writing a full-out function declaration or even assigning the lambda function to a local variable.
# 
# Lambda functions are very Pythonic and let us to write simple functions on the fly.
# For example, we could use a lambda function to sort `strings` by the number of unique letters.

# In[116]:


strings = ['foo', 'card', 'bar', 'aaaa', 'abab']


# In[117]:


strings.sort()
strings


# In[118]:


strings.sort(key=len)
strings


# In[119]:


strings.sort(key=lambda x: x[-1]) # this lambda function slices the last character in each string i
strings


# How can I sort by the *second* letter in each string?

# In[120]:


strings[2][1]


# In[121]:


strings.sort(key=lambda x: x[1]) # this lambda function slices the second character in each string i
strings


# ## Practice

# ***Practice:***
# Swap the values assigned to `a` and `b` using a third variable `c`.

# In[122]:


a = 1


# In[123]:


b = 2


# In[124]:


c = a


# In[125]:


a = b


# In[126]:


b = c


# In[127]:


del c


# In[128]:


print(f'a is {a} and b is {b}')


# ***Practice:***
# Use tuple-unpacking to swap the values assigned to `a` and `b` ***without*** using a third variable.

# In[129]:


a = 1


# In[130]:


b = 2


# In[131]:


a, b = b, a


# In[132]:


print(f'a is {a} and b is {b}')


# ***Practice:*** 
# Create a list of integers from 1 to 100 using `range()` and `list()` named `l1`.

# In[133]:


l1 = list(range(1, 101))


# ***Practice:*** 
# Slice `l1` to create a list of integers from 60 to 50 (inclusive) named `l2`.

# In[134]:


l2 = l1[59:48:-1]


# In[135]:


l1[-41:-52:-1] 


# In[136]:


l1[49:60][::-1]


# If we do not want to slice `l1`, prettier solutions are possible:

# In[137]:


list(range(60, 49, -1))


# ***Practice:***
# Create a list of odd integers from 1 to 21 named `l3`.

# In[138]:


l3 = list(range(1, 22, 2))


# ***Practice:***
# Use a list comprehension to create a list `l4` that contains the squares of integers from 1 to 100.

# In[139]:


l4 = [i**2 for i in range(1, 101)]


# ***Practice:***
# Use a list comprehension to create a list `l5` that contains the squares of ***odd*** integers from 1 to 100.

# In[140]:


l5 = [i**2 for i in range(1, 101, 2)]


# In[141]:


print(l5, end=' ')


# In[142]:


print([i**2 for i in range(1, 101) if i%2==1], end=' ')


# ***Practice:***
# Use a lambda function to sort `strings` by the last letter.

# In[143]:


strings.sort(key=lambda x: x[-1])
strings


# You can find many more Python coding challenges at <www.hackerrank.com> and <www.leetcode.com>.
# These challenges are fun but not necessary for our course because we will spend our time using higher-level features.
