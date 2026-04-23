import time

# pi = 4*(1-1/3+1/5-...)

def pi(n_terms):
    pi_approximation = 0.0
    sign = 1  # This will alternate between 1 and -1

    for i in range(n_terms):
        term = sign * (1 / (2 * i + 1))
        pi_approximation += term
        sign *= -1  # Alternate the sign

    return 4 * pi_approximation


start_time = time.time()

pi = pi(100_000_000)

print(f"The result is {pi}")

end_time = time.time()
print(f"It took {end_time-start_time:.2f} seconds to compute")