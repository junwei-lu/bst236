"""
Compute an approximation of pi using the series:
pi = 4 * (1 - 1/3 + 1/5 - 1/7 + ...) n . . . . . 

Parameters:
n_terms (int): The number of terms to include in the series.

Returns:
float: The approximation of pi.
"""

import numba
import time

def pi(n_terms):
    pi_approximation = 0.0
    sign = 1  # This will alternate between 1 and -1

    for i in range(n_terms):
        term = sign * (1 / (2 * i + 1))
        pi_approximation += term
        sign *= -1  # Alternate the sign

    return 4 * pi_approximation

@numba.jit
def pi_jit(n_terms):
    pi_approximation = 0.0
    sign = 1  # This will alternate between 1 and -1

    for i in range(n_terms):
        term = sign * (1 / (2 * i + 1))
        pi_approximation += term
        sign *= -1  # Alternate the sign

    return 4 * pi_approximation


def profiling(func, n = 100_000_000):
    start_time = time.time()
    result = func(n)
    end_time = time.time()
    print(f"Time taken for {func.__name__} with {n} terms: {end_time - start_time} seconds")
    return result

# Example usage:
if __name__ == "__main__":
    pi_jit(0)
    profiling(pi)
    profiling(pi_jit)