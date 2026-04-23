from multiprocessing import Pool, cpu_count, current_process
import os

def worker(number):
    """Function to be executed by each process"""
    print(f"Process ID: {os.getpid()}, Process Name: {current_process().name}, Number: {number}")
    return number  # Optional: return a value if you need results

def main():
    # Get the number of CPU cores (leave one core free for system processes)
    num_processes = max(1, cpu_count() - 1)
    print(f"Using {num_processes} processes")
    
    # Data to process
    numbers = range(10)  # Example: process numbers 0-9

    # Using Pool context manager for automatic resource management
    with Pool(processes=num_processes) as pool:
        # Map the worker function to the data
        results = pool.imap(worker, numbers)
        
        # Wrong: Only prints the iterator object
        #print(f"Only iterator results: {results}")
         # Consume the iterator to see the results
        results_list = list(results)
        print(f"Real results: {results_list}")


if __name__ == '__main__':
    main() 