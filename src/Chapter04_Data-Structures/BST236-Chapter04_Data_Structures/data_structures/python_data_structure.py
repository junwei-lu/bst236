#%% Imports and Setup
import numpy as np
import pandas as pd
from collections import deque
import timeit
from functools import wraps


def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        print(f"{func.__name__} took {(end_time - start_time):.4f} seconds")
        return result
    return wrapper

#%% 1. Lists in Python
print("1. Lists in Python")
print("-----------------")

# Basic list operations
my_list = [1, 2, 3, 4, 5]
print("Original list:", my_list)
my_list.append(6)
print("After append/extend:", my_list)
print("Element at index 2:", my_list[2])
# search for element 5
print("Index of element 5:", my_list.index(5))
# delete element at index 0
del my_list[0]


#%% List Slicing Examples
print("\nList Slicing Examples:")
sample_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print("Original list:", sample_list)
# list[start:stop:step]
print("First 3 elements [0:3]:", sample_list[0:3])
print("Last 3 elements [-3:]:", sample_list[-3:])
print("Every second element [::2]:", sample_list[::2])
print("Reverse list [::-1]:", sample_list[::-1])
print("Elements from index 2 to 7 [2:7]:", sample_list[2:7])


#%% List as a Stack
print("\nList as a Stack:")
my_stack = []
my_stack.append(1)
my_stack.append(2)
print("Stack after push:", my_stack)
print("Stack after pop:", my_stack.pop())

#%% zip for loop
print("\nZip Example:")
list1 = [1, 2, 3, 4, 5]
list2 = ['a', 'b', 'c', 'd', 'e']
print("List 1:", list1)
print("List 2:", list2)
print("\nIterating over zipped lists:")
for num, letter in zip(list1, list2):
    print(f"Number: {num}, Letter: {letter}")
for num, letter in enumerate(list2):
    print(f"Index: {num}, Letter: {letter}")

#%% List Performance Comparison
print("\nComparing list operations with and without loop:")

@timer_decorator
def square_with_loop(size=1000):
    numbers = list(range(size))
    squared = []
    for num in numbers:
        squared.append(num ** 2)
    return squared

@timer_decorator
def square_with_comprehension(size=1000):
    numbers = list(range(size))
    return [num ** 2 for num in numbers]

@timer_decorator
def square_with_comprehension_even(size=1000):
    numbers = list(range(size))
    return [num ** 2 for num in numbers if num % 2 == 0]

# Compare list operations
n = 10000000
_ = square_with_loop(n)
_ = square_with_comprehension(n)
_ = square_with_comprehension_even(n)

#%% Map, Filter, Any Examples
print("\nMap, Filter, Any Examples:")
print("-------------------------")

# Map example - apply function to every element
numbers = [1, 2, 3, 4, 5]
def square(x):
    return x**2
squared = list(map(square, numbers))
print("Original numbers:", numbers)
print("After map (squared):", squared)

@timer_decorator
def square_with_map(size=1000):
    numbers = list(range(size))
    return list(map(lambda x: x**2, numbers))

@timer_decorator 
def square_with_for(size=1000):
    numbers = list(range(size))
    result = []
    for num in numbers:
        result.append(num**2)
    return result

# Compare performance
n = 10000000
print("\nTiming with size =", n)
_ = square_with_map(n)
_ = square_with_for(n)


# Multiple iterables with map
list1 = [1, 2, 3]
list2 = [10, 20, 30] 
def add(x, y):
    return x + y
sums = list(map(add, list1, list2))
print("\nMapping over two lists:")
print("List 1:", list1)
print("List 2:", list2)
print("Mapped sums:", sums)

# Filter example - keep elements that satisfy condition
numbers = range(-5, 6)
positives = list(filter(lambda x: x > 0, numbers))
print("\nOriginal range:", list(numbers))
print("After filter (positives only):", positives)

# Any example - check if any element satisfies condition
numbers = [1, 3, 5, 6, 7, 9]
has_even = any(num % 2 == 0 for num in numbers)
print("\nNumbers:", numbers)
print("Contains at least one even number:", has_even)

all_positive = all(num > 0 for num in numbers)
print("All numbers are positive:", all_positive)

#%% 2. Deque Operations
print("\n2. Deque (Double-ended Queue)")
print("-----------------------------")

# We typically use deque as queue
my_deque = deque([1, 2, 3])
print("Original deque:", my_deque)
my_deque.append(4)
print("After append operations:", my_deque)
my_deque.popleft()
print("After pop operations:", my_deque)

# Iterating over deque
print("\nIterating over deque:")
for item in my_deque:
    print(item, end=' ')
print() 

#%% 3. Dictionary Operations
print("\n3. Dictionary (Hash Table)")
print("-------------------------")

# Basic dictionary operations
hmap = {'a': 1, 'b': 2, 'c': 3}
print("Original dictionary:", hmap)
# Add operation
# Add key-value pair (key, value) to the hash table
hmap['d'] = 4
print("After adding 'd':", hmap)
print("Value of 'b':", hmap['b'])
print("Value of 'e':", hmap['e'])
print("Value of 'e':", hmap.get('e'))


# Dictionary comprehension
squared_dict = {k: v**2 for k, v in hmap.items()}
print("Squared dictionary:", squared_dict)

# The defaultdict works like a regular dictionary but allows you to specify a default value type for keys that are accessed but do not yet exist in the dictionary.
from collections import defaultdict
counts = defaultdict(int)  # Default value is 0
counts['apple'] += 1
counts['banana'] += 1
print(counts) 

#%% Traverse a dictionary
# Traverse hash table
# Traverse key-value pairs key->value
for key, value in hmap.items():
    print(key, "->", value)
# Traverse keys only
for key in hmap.keys():
    print(key)
# Traverse values only
for value in hmap.values():
    print(value)

#%% Dictionary Performance Comparison
print("\nComparing dictionary vs list for lookups:")

@timer_decorator
def lookup_in_list(items):
    target = len(items) - 1
    for item in items:
        if item == target:
            break
    return item


@timer_decorator
def lookup_in_dict(items):
    target = len(items) - 1
    return items[target]

# Compare lookups
n = 10000000
items_list = list(range(n))
items_append = []
for i in range(n):
    items_append.append(i)
items_dict = {i: i for i in range(n)}
# List search time: O(n)
_ = lookup_in_list(items_list)
# Dict search time: O(1)
_ = lookup_in_dict(items_dict)
# List non-static search time: O(n)
_ = lookup_in_list(items_append)

#%% the space complexity of the list is O(n)
import sys
print(f"List size: {sys.getsizeof(items_list):,} bytes")
print(f"Dict size: {sys.getsizeof(items_dict):,} bytes")

# Use Lists when:
#  * Order matters
#  * Memory efficiency is important
#  * You need integer indexing and slicing
# Use Dictionaries when:
#  * You need fast lookups by key
#  * Order doesn't matter 
#  * You need to associate data with non-integer keys
#  * Performance is more important than memory usage


#%% 4. NumPy Array Operations
print("\n4. NumPy Arrays")
print("--------------")

# Basic numpy operations
np_array = np.array([1, 2, 3, 4, 5])
print("Original array:", np_array)
print("Array squared:", np_array ** 2)
print("Array mean:", np_array.mean())

even_arr = np_array[np_array % 2 == 0]
print("Even numbers:", even_arr)  

# NumPy Array Slicing Examples
print("\nNumPy Array Slicing Examples:")
sample_array = np.arange(10)
print("Original array:", sample_array)
print("First 3 elements [0:3]:", sample_array[0:3])
print("Last 3 elements [-3:]:", sample_array[-3:])
print("Every second element [::2]:", sample_array[::2])
print("2D array slicing:")
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("2D array:\n", array_2d)
print("First two rows, all columns [0:2,:]:\n", array_2d[0:2,:])
print("All rows, last two columns [:,1:]:\n", array_2d[:,1:])

#%% Matrix Operations
x = np.arange(12)
x = x.reshape(3, 4)
print("Original array:\n", x)
print("Mean of each column:", np.mean(x, axis=0))
print("Subtract mean of each column:\n", x-np.mean(x, axis=0))

#%% NumPy Performance Comparison
print("\nComparing numpy vs list for calculations:")

@timer_decorator
def calculate_with_list(size=1000):
    numbers = list(range(size))
    return [num ** 2 for num in numbers]

@timer_decorator
def calculate_with_numpy(size=1000):
    numbers = np.arange(size)
    return numbers ** 2

# Compare calculations
n = 10_000_000
_ = calculate_with_list(n)
_ = calculate_with_numpy(n)

print(f"List size: {sys.getsizeof(list(range(n))):,} bytes")
print(f"Numpy array size: {sys.getsizeof(np.arange(n)):,} bytes")

# List vs NumPy Array
# Use Lists for:
# * General purpose programming
# * Small datasets
# * Mixed data types
# Use NumPy Arrays for:
# * Scientific computing
# * Vectorized operations

#%% 5. Pandas DataFrame Operations
print("\n5. Pandas DataFrame")
print("-----------------")

# Basic DataFrame operations
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})
print("Original DataFrame:")
print(df)
print("\nColumn 'A':", df['A'].values)
print("Mean of each column:")
print(df.mean())

#%% Pandas DataFrame Slicing Examples
print("\nPandas DataFrame Slicing Examples:")
sample_df = pd.DataFrame({
    'A': range(10),
    'B': range(10, 20),
    'C': range(20, 30)
})
print("Original DataFrame:")
print(sample_df)
print("\nFirst 3 rows using .iloc[0:3]:")
print(sample_df.iloc[0:3])
print("\nFirst 2 columns using .loc[:,['A','B']]:")
print(sample_df.loc[:,['A','B']])
print("\nRows where A > 5:")
print(sample_df[sample_df['A'] > 5])
print("\nRows 1-3, columns A and C:")
print(sample_df.loc[1:3, ['A','C']])

#%% Pandas Apply
print("\nPandas Apply:")
# Create sample DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})
print("Original DataFrame:")
print(df)

# Apply exp to each element using apply
print("\nAfter applying exp to each element:")
print(df.apply(np.exp))

# Apply mean to specific column
print("\nApply mean to each column:")
print(df.apply(np.mean))
print("\nApply mean to each row:")
print(df.apply(np.mean, axis=1))


#%% Pandas Performance Comparison
print("\nComparing pandas vs loop for data analysis:")

size=1_000_000
data = {'A': list(range(size)), 'B': list(range(size))}
df = pd.DataFrame(data)

@timer_decorator
def calculate_means_with_loop():
    column_means = {}
    for col in data:
        column_means[col] = sum(data[col]) / len(data[col])
    return column_means

@timer_decorator
def calculate_means_with_pandas():
    return df.mean()

# Compare mean calculations
_ = calculate_means_with_loop()
_ = calculate_means_with_pandas()

#%% Group by 
# Create a sample DataFrame
df = pd.DataFrame({
    'Category': ['Fruit', 'Fruit', 'Vegetable', 'Vegetable', 'Fruit', 'Vegetable'],
    'Name': ['Apple', 'Banana', 'Carrot', 'Broccoli', 'Orange', 'Spinach'],
    'Price': [1.2, 0.5, 0.8, 1.5, 0.7, 1.0],
    'Quantity': [10, 15, 20, 12, 8, 25]
})

# Group by Category and calculate mean price
category_avg_price = df.groupby('Category')['Price'].mean()
print("Average Price by Category:")
print(category_avg_price)

# Multiple Aggregations
category_summary = df.groupby('Category').agg({
    'Price': 'mean',    # Average price
    'Quantity': 'sum',  # Total quantity
    'Name': 'count'     # Number of items
})
print("\nCategory Summary:")
print(category_summary)

# Custom Aggregation Functions
def price_range(x):
    return x.max() - x.min()

custom_agg = df.groupby('Category').agg({
    'Price': [
        ('Avg Price', 'mean'),
        ('Price Range', price_range)
    ],
    'Quantity': [
        ('Total Quantity', 'sum'),
        ('Avg Quantity', 'mean')
    ]
})
print("\nCustom Category Aggregation:")
print(custom_agg)