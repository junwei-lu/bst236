# Python Lists


Lists are one of the most versatile and commonly used data structures in Python. They allow you to store and manipulate collections of items efficiently. 

## Basic Operations

Lists in Python can be created easily by enclosing items in square brackets:

```python
my_list = [1, 2, 3, 4, 5]
my_list = list(range(1, 6))
my_list = [0] * 10
```


**Insert**: The `append()` method adds a single element to the end of the list:

```python
my_list.append(6)
print("After append:", my_list)
# Result: [1, 2, 3, 4, 5, 6]
```



**Search**: The `index()` method returns the index of the first occurrence of a specified element:

```python
print("Index of element 5:", my_list.index(5))
# Result: 4
```

**Delete**: The `del` statement removes an element from the list by its index:
```python
del my_list[0]
```

## List Slicing

List slicing allows you to extract portions of a list using a powerful and flexible syntax.


The general syntax is `list[start:end:step]`

Extract First 3 Elements
```python
sample_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print("First 3 elements [0:3]:", sample_list[0:3])
# Result: [0, 1, 2]
```
Extract Last 3 Elements
```python
print("Last 3 elements [-3:]:", sample_list[-3:])
# Result: [7, 8, 9]
```

Every Second Element
```python
print("Every second element [::2]:", sample_list[::2])
# Result: [0, 2, 4, 6, 8]
```

Reverse the List
```python
print("Reverse list [::-1]:", sample_list[::-1])
# Result: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
```


## List for Loops

### List Comprehensions 

You should use list comprehensions instead of traditional loops.

```python
# Traditional loop method
def square_with_loop(size=1000):
    numbers = list(range(size))
    squared = []
    for num in numbers:
        squared.append(num ** 2)
    return squared

# List comprehension method
def square_with_comprehension(size=1000):
    numbers = list(range(size))
    return [num ** 2 for num in numbers]

# List comprehension with condition
def square_with_comprehension_even(size=1000):
    numbers = list(range(size))
    return [num ** 2 for num in numbers if num % 2 == 0]
```

### List Functional Programming


`map()` applies a function to every item in an iterable:

```python
# Squaring numbers
numbers = [1, 2, 3, 4, 5]
# Define a squaring function
def square(x):
    return x**2
# Using map to square numbers
squared = list(map(square, numbers))
```
`filter()` creates a list of elements that satisfy a condition:

```python
# Filter positive numbers
numbers = range(-5, 6)
positives = list(filter(lambda x: x > 0, numbers))
```

 `any()` and `all()` check conditions across list elements:

```python
numbers = [1, 3, 5, 6, 7, 9]
# Check if any number is even
has_even = any(num % 2 == 0 for num in numbers)
# Check if all numbers are positive
all_positive = all(num > 0 for num in numbers)
```

 `zip()` function allows you to combine multiple lists element-wise:

```python
list1 = [1, 2, 3, 4, 5]
list2 = ['a', 'b', 'c', 'd', 'e']
# Iterating over zipped lists
for num, letter in zip(list1, list2):
    print(f"Number: {num}, Letter: {letter}")
```

`enumerate()` adds a counter to an iterable:

```python
for index, letter in enumerate(list2):
    print(f"Index: {index}, Letter: {letter}")
```



## Key Performance Insights

1. **List Comprehensions**
      - Generally faster than traditional loops
      - More readable and concise
      - Can include conditional filtering

2. **Functional Methods**
      - Functional methods are not always faster than explicit loops
      - Sometimes readability trumps minor performance gains
