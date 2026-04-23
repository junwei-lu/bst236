import time
#import tqdm

# read the txt file measurements.txt
with open("measurements.txt", "r") as file:
    measurements = file.readlines()

# parse the measurements into a list of tuples (city, temperature)
measurements = [line.strip().split(';') for line in measurements]
measurements = [(city, float(temp)) for city, temp in measurements]

# Measure time for calculating stats per city
start_time = time.time()

# Create dictionary to store temperatures by city
city_temps = {}
for city, temp in measurements:
    if city not in city_temps:
        city_temps[city] = []
    city_temps[city].append(temp)

# Calculate min, max, avg for each city
city_stats = {}
for city, temps in city_temps.items():
    city_stats[city] = {
        'min': min(temps),
        'max': max(temps),
        'avg': sum(temps) / len(temps)
    }

stats_time = time.time() - start_time

# Print results
print("City statistics:")
for city in sorted(city_stats.keys()):
    stats = city_stats[city]
    print(f"{city}: min={stats['min']:.1f}, max={stats['max']:.1f}, avg={stats['avg']:.1f}")
print(f"\nComputed in {stats_time:.6f} seconds")