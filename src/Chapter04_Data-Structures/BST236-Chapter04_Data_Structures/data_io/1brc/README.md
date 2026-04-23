# Big Data Processing in Python

In this folder, we will use the one billion row challenge data set to test the performance of different Python libraries. You can find how different python packages perform on this task in different time.

We expect you to get to know the following libraries:
- Pandas: The most popular library for data manipulation and analysis.
- Polars: A high-performance DataFrame library that is similar to Pandas but faster for certain operations.
- DuckDB: An in-memory database that is optimized for analytical queries.
- Numba: A just-in-time compiler for Python that can speed up numerical computations.
- Pypy: A Python interpreter that can speed up Python code.
- Multiprocessing: A library for parallel execution of Python code.

## Python Timing

```bash
python -m cProfile your_script.py
viztracer your_script.py
viztracer --include_files ./ --run my_script.py
```


