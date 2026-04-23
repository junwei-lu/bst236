# 1BRC: One Billion Row Challenge in Python

Python implementation of Gunnar's 1 billion row challenge:
- https://www.morling.dev/blog/one-billion-row-challenge
- https://github.com/gunnarmorling/1brc

## Creating the measurements file with 1B rows

First install the Python requirements:
```shell
python3 -m pip install -r requirements.txt
```

The script `createMeasurements.py` will create the measurement file:
```text
Hamburg;12.0
Bulawayo;8.9
Palembang;38.8
St. John's;15.2
Cracow;12.6
...
```
The program should print out the min, mean, and max values per station, alphabetically ordered like so:
```text
{Abha=5.0/18.0/27.4, Abidjan=15.7/26.0/34.1, Abéché=12.1/29.4/35.6, Accra=14.7/26.4/33.1, Addis Ababa=2.1/16.0/24.3, Adelaide=4.1/17.3/29.7, ...}
```


