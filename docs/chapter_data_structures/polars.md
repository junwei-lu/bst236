# Polars
Polars is a modern DataFrame library that offers significant performance advantages over Pandas, especially for large datasets. We compare the basic operations between Polars and Pandas.

## Basic DataFrame Creation and Reading

Pandas

```python
import pandas as pd

# Read CSV
df = pd.read_csv("data.csv", sep=";", header=None, names=['city', 'temp'])

# Process in chunks for large files
reader = pd.read_csv("data.csv", sep=';', chunksize=1000000)
for chunk in reader:
    # Process chunk
    pass
```

Polars

```python
import polars as pl

# Read CSV (lazy evaluation)
df = pl.scan_csv(
    "data.csv",
    separator=";",
    has_header=False,
    new_columns=["city", "temp"]
)
```
### Key Differences

- Lazy Evaluation:
    - Polars uses lazy evaluation by default with scan_csv() and .lazy()
    - Operations are only executed when you call .collect()
    - This allows Polars to optimize the query plan
- Streaming:

```python
# Polars streaming for large datasets
results = (
    df.lazy()
    .group_by("city")
    .agg([
        pl.col("temp").min(),
        pl.col("temp").max(),
        pl.col("temp").mean()
    ])
    .collect(streaming=True)  # Enable streaming
)
```

### Performance Benefits

- Better memory efficiency
- Faster processing for large datasets
- Built-in parallel processing
- Column-oriented design

### When to Use Polars

- Large datasets that don't fit in memory
- Performance-critical applications
- When you need parallel processing
- Modern data pipeline development

