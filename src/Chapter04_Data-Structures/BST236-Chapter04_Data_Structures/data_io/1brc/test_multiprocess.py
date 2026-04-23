import pandas as pd
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os

def process_chunk(chunk):
    """Process a single chunk of data"""
    try:
        # Convert temperature column to float if it's not already
        chunk['temp'] = pd.to_numeric(chunk['temp'], errors='coerce')
        # Drop any rows where conversion failed
        chunk = chunk.dropna()
        
        # Group by 'city' and calculate min, max, and mean for the chunk
        results = chunk.groupby('city')['temp'].agg(['min', 'max', 'mean']).rename(columns={
            'min': 'temperature_min',
            'max': 'temperature_max',
            'mean': 'temperature_mean'
        })
        return results
    except Exception as e:
        print(f"Error processing chunk: {e}")
        print(f"Chunk head: {chunk.head()}")
        return pd.DataFrame()  # Return empty DataFrame on error

def process_data(file_path, chunk_size=1000000):    
    # Get the number of CPU cores (leave one core free for system processes)
    num_processes = max(1, cpu_count() - 1)
    print(f"Using {num_processes} processes")

    # Check first few lines of the file
    print("Checking file content:")
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 5:  # Print first 5 lines
                print(f"Line {i+1}: {line.strip()}")
            else:
                break

    # Initialize an empty DataFrame to accumulate results
    accumulated_results = pd.DataFrame()

    # Initialize reader object to get total size for progress bar
    total_size = os.path.getsize(file_path)
    total_chunks = total_size // (chunk_size * 100)  # Rough estimate of number of chunks

    # Initialize the process pool
    with Pool(processes=num_processes) as pool:
        # Create a reader that yields chunks
        reader = pd.read_csv(
            file_path, 
            sep=';', 
            header=None,  # No header in file
            names=['city', 'temp'],
            dtype={'city': str},  # Only specify city dtype
            chunksize=chunk_size,
            skip_blank_lines=True,
            on_bad_lines='skip'
        )

        # Process chunks in parallel
        results = []
        chunk_iterator = pool.imap(process_chunk, reader)
        for chunk_result in tqdm(chunk_iterator, total=total_chunks):
            if not chunk_result.empty:
                results.append(chunk_result)

    if not results:
        raise ValueError("No valid results were processed")

    # Combine all results
    accumulated_results = pd.concat(results)

    # Final aggregation to ensure city stats are correct across all chunks
    final_results = accumulated_results.groupby(accumulated_results.index).agg({
        'temperature_min': 'min',
        'temperature_max': 'max',
        'temperature_mean': 'mean'
    })
    return final_results

if __name__ == '__main__':
    # Specify your file path
    file_path = 'measurements.txt'
    start_time = time.time()
    
    try:
        city_stats = process_data(file_path)
        
        # Open the file in write mode
        with open("result_parallel.txt", "w") as file:
            # Write final results to the file
            file.write("{")
            for i, (city, row) in enumerate(city_stats.iterrows()):
                file.write(
                    f"{city}={row['temperature_min']:.1f}/{row['temperature_mean']:.1f}/{row['temperature_max']:.1f}"
                )
                if i < len(city_stats) - 1:
                    file.write(", ")
            file.write("}")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    
    except Exception as e:
        print(f"Error processing file: {e}")

