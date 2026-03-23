# Pandas DataFrame

<a href="https://colab.research.google.com/github/junwei-lu/bst236/blob/main/codes/chapter04_data_structures.ipynb#scrollTo=cb6147c3a84a" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; background: linear-gradient(135deg, #2e7d32 0%, #66bb6a 100%); color: white; border-radius: 6px; text-decoration: none; font-size: 0.85em; font-weight: 600;">▶ Try in Colab</a>

Pandas is a powerful library for data manipulation and analysis. It provides data structures and operations for manipulating numerical tables and time series.

### Basic Operations

```python
import pandas as pd

# Basic DataFrame Creation
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

# Accessing Columns
column_a = df['A'].values
column_means = df.mean()
```

**Search**: 
```python
# Find index of specific value
index_of_value = df[df['A'] == 3].index[0]
# Find rows meeting a condition
filtered_rows = df[df['A'] > 3]
```

**Delete**: The `drop()` method removes rows or columns from the DataFrame:

```python
df.drop(columns=['A']) # Drop column A
df_dropped = df.drop([0, 2])  # Drop rows at index 0 and 2
```

### Slicing 

```python
# Create Sample DataFrame
sample_df = pd.DataFrame({
    'A': range(10),
    'B': range(10, 20),
    'C': range(20, 30)
})

# Slicing Techniques
# First 3 rows
first_three_rows = sample_df.iloc[0:3]

# Select specific columns
selected_columns = sample_df.loc[:, ['A', 'B']]

# Conditional Filtering
filtered_rows = sample_df[sample_df['A'] > 5]

# Complex Slicing
specific_slice = sample_df.loc[1:3, ['A', 'C']]
```

### Apply Function

```python
# Apply Functions to DataFrame
# Exponential of each element
exp_df = df.apply(np.exp)

# Calculate Mean
column_means = df.apply(np.mean)
row_means = df.apply(np.mean, axis=1)
```

## Performance Considerations

### NumPy vs Lists
- Use Lists for:
  - General-purpose programming
  - Small datasets
  - Mixed data types

- Use NumPy Arrays for:
  - Scientific computing
  - Vectorized operations
  - Large numerical computations

### Memory Efficiency
```python
# Memory Comparison
list_size = sys.getsizeof(list(range(10000000)))
numpy_size = sys.getsizeof(np.arange(10000000))
```
