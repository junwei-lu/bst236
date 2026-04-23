import time
from tqdm import tqdm

def read_measurements(filename: str):
    """Read the measurements from a file and return the max, min and average temperatures of each city."""
    city_stats = {}
    with open(filename, "r") as file:
        for line in tqdm(file, desc="Processing data"):
            city, temp = line.strip().split(';')
            if city in city_stats:
                stats = city_stats[city]
                stats['count'] += 1
                stats['total'] += temp
                if temp < stats['min']:
                    stats['min'] = temp
                if temp > stats['max']:
                    stats['max'] = temp
            else:
                city_stats[city] = {
                    'min': temp,
                    'max': temp,
                    'total': temp,
                    'count': 1
                }
    # Calculate mean from total and count
    for city, stats in city_stats.items():
        stats['mean'] = stats['total'] / stats['count']

    return city_stats

if __name__ == "__main__":

    start_time = time.time()
    city_stats = read_measurements('measurements.txt')
    end_time = time.time()

    print('Helsinki',city_stats['Helsinki'])
    print('Guatemala City,',city_stats['Guatemala City'],'\n')

    print(f"Time elapsed: {end_time - start_time:.2f} seconds") 


            
