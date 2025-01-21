# Python 101

This is a basic tutorial for Python. We expect you to know the following materials for understanding the materials in the class. It covers the fundamentals of Python programming. For more topics, you can refer to the [official Python tutorial](https://wiki.python.org/moin/BeginnersGuide).

## Table of Contents
- [Python 101](#python-101)
  - [Table of Contents](#table-of-contents)
  - [Getting Started ](#getting-started-)
  - [Basic Commands ](#basic-commands-)
  - [Data Types ](#data-types-)
  - [Data Structures ](#data-structures-)
    - [Lists](#lists)
    - [Sets](#sets)
    - [Dictionaries](#dictionaries)
  - [Control Flow ](#control-flow-)
    - [Conditional Statements](#conditional-statements)
    - [For Loops](#for-loops)
    - [While Loops](#while-loops)
  - [Functions ](#functions-)
    - [Lambda Functions](#lambda-functions)
  - [Advanced Topics ](#advanced-topics-)
    - [List Comprehensions](#list-comprehensions)
    - [Object-Oriented Programming](#object-oriented-programming)
  - [Data Analysis with Pandas and Scikit-learn ](#data-analysis-with-pandas-and-scikit-learn-)

## Getting Started <a name="getting-started"></a>

Before diving into Python programming, it's important to know which version you're using. Python has two major versions (2 and 3) with significant differences. Here's how to check your Python version:

```python
import sys
print("Python version")
print(sys.version)
print("Version info.")
print(sys.version_info)
```
We suggest using Python 3 and following the [installation guide](./installation.md#python).

## Basic Commands <a name="basic-commands"></a>

Python's syntax is clean and readable. Here are some basic commands:

```python
# Print text
print('Hello World!')

# Perform calculations
print(3 + 3)

'''
You can write multi-line
comments using triple quotes
'''
```

## Data Types <a name="data-types"></a>

Python has several built-in data types. You can check the type of any value using the `type()` function:

```python
# Integer
print(type(3))  # <class 'int'>

# Float
print(type(3.0))  # <class 'float'>

# String
print(type('hello'))  # <class 'str'>

# Boolean
print(type(True))  # <class 'bool'>

# Boolean expression
print(type(3 == 3))  # <class 'bool'>
```

## Data Structures <a name="data-structures"></a>

### Lists
Lists are one of Python's most versatile data structures. Key features:
- Mutable (can be modified)
- Can contain multiple data types
- Support indexing and slicing
- Zero-based indexing

```python
# Creating lists
foo = [1, 2, 3]
bar = ['a', 'b', 'c']
baz = [16, 'apple', 17.4, True]

# Accessing elements
print(foo[0])  # First element
print(bar[1:3])  # Slice from index 1 to 2

# Modifying lists
foo[1] = 'a'
```

### Sets
Sets are unique collections of elements. Key features:
- Unordered
- Cannot be indexed or sliced
- Elements must be unique
- Useful for removing duplicates from lists

```python
# Creating sets
new_set = {1, 2, 1, 2, 3, 4, 1}  # Duplicates are automatically removed
print(new_set)  # {1, 2, 3, 4}

# Converting list to set and back (removes duplicates)
l = [1, 1, 2, 3]
print(list(set(l)))  # [1, 2, 3]
```

### Dictionaries
Dictionaries store key-value pairs. Key features:
- Fast lookup by key
- Keys must be unique
- Values can be any data type
- Support nested structures

```python
state_dict = {
    'Pennsylvania': 20,
    'Arizona': 11,
    'Georgia': 16,
    'Nevada': 6
}

print('Georgia has', state_dict['Georgia'], 'Electoral Votes')
```

## Control Flow <a name="control-flow"></a>

### Conditional Statements
Python uses indentation to define code blocks:

```python
num1 = 5
num2 = 100

if num1 > num2:
    print(num1, 'is greater than', num2)
elif num1 < num2:
    print(num1, 'is less than', num2)
else:
    print(num1, 'is equal to', num2)
```

### For Loops
Python's for loops are versatile and can iterate over many types of sequences:

```python
# Range-based loop
for i in range(1, 5):
    print(i)

# Iterating over a list
foo = ['a', 'b', 'c', 'd']
for letter in foo:
    print(letter)

# Iterating over dictionary
for key, val in state_dict.items():
    print(key, val)
```

### While Loops
While loops continue until a condition is met:

```python
# Calculate factorial
num = 5
fac = 1
while num > 1:
    fac *= num
    num -= 1
```

## Functions <a name="functions"></a>

Functions in Python are defined using the `def` keyword:

```python
def sum_fun(a, b):
    '''
    This function takes two input numbers
    and returns their sum

    Input: a (int/float), b(int/float)
    Output: my_sum(int/float)
    '''
    my_sum = a + b
    return my_sum

print(sum_fun(5, 7))
```

### Lambda Functions
Lambda functions are anonymous, inline functions:

```python
# Filter even numbers using lambda
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = list(filter(lambda x: x % 2 == 0, my_list))
```

## Advanced Topics <a name="advanced-topics"></a>

### List Comprehensions
List comprehensions provide a concise way to create lists:

```python
# Traditional way
foo = []
bar = [1, 17, 8, 83, 26, 11, 14, 92, 37]
for num in bar:
    if num % 2 == 0:
        foo.append(num)

# List comprehension
foo = [num for num in bar if num % 2 == 0]
```

### Object-Oriented Programming
Python is an object-oriented language. Here's a simple class example:

```python
class Dog:
    def __init__(self, name):
        self.name = name
        self.breed = ''
        self.age = 0
        self.fed_status = 'Not fed'

    def have_birthday(self):
        self.age += 1

    def feed(self, food):
        self.fed_status = f'{self.name} has been fed with {food}'

# Create and use an instance
my_dog = Dog('Spot')
my_dog.feed('pizza')
my_dog.have_birthday()
```

## Data Analysis with Pandas and Scikit-learn <a name="data-analysis"></a>

Python is powerful for data analysis using libraries like Pandas and Scikit-learn:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm

# Read data
data = pd.read_csv('data.csv')

# Data preprocessing
X = data.drop(columns=['target'])
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Create and train model
model = svm.SVC()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

