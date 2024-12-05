# Coding in style

Please check the complete Python style guide [PEP 8](https://peps.python.org/pep-0008/) or [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). You can also check the detailed [R style guide](https://google.github.io/styleguide/Rguide.html).
Here we just want to share some basic coding style principles.
### Naming objects

Names should be descriptive and self-documenting:

```python
# Bad example - not self-documenting
def func(y):
    # z is gravitational force
    z = 9.81
    # y is time in seconds
    y = 5
    # x is displacement
    x = 1/2 * z * y**2
    return x

# Good example - self-documenting
def compute_displacement(time_in_seconds):
    gravitational_force = 9.81
    displacement = 1/2 * gravitational_force * time_in_seconds**2
    return displacement

# Even better - with parameters
def compute_displacement(time_in_seconds, 
                       gravitational_force=9.81):
    displacement = 1/2 * gravitational_force * time_in_seconds**2
    return displacement
```

Use either `snake_case` or `camelCase` but make it consistent:

```python
# snake_case (Python preferred style)
def calculate_average(number_list):
    return sum(number_list) / len(number_list)

# camelCase (less common in Python)
def calculateAverage(numberList):
    return sum(numberList) / len(numberList)
```

Python name conventions: 

- Module names should use `snake_case`.
- Class names should use `CapWords`.
- Function and variable names should use `snake_case`.
- Constant names should use `UPPERCASE`.
- Private names should use a leading underscore, e.g., `_private_variable`.
- Protected names should use a double leading underscore, e.g., `__protected_variable`.

### Documentation

Use docstrings to explain what the module, class, or function does.

```python
# Bad example
def f(x, y):
    a = x + y  # What does 'a' represent?
    return a

# Good example
def calculate_total_price(item_price, tax_rate):
    """Calculate final price including tax.
    
    Args:
        item_price (float): Original price of the item
        tax_rate (float): Tax rate as a decimal (e.g., 0.1 for 10%)
    
    Returns:
        float: Total price including tax
    """
    total_price = item_price * (1 + tax_rate)
    return total_price
```

### Spacing and Operators

```python
# Bad spacing
x=y+z
if x>0:print(x)

# Good spacing
x = y + z
if x > 0:
    print(x)
```

### Indentation
Use 4 spaces per indentation level. Avoid mixing tabs and spaces:

```python
# Good indentation
def process_data():
    if condition:
        do_something()
        for item in items:
            process_item()
    
# Bad indentation (mixing spaces)
def process_data():
   if condition:
      do_something()
        for item in items:
         process_item()
```

### Maximum Line Length
Limit all lines to a maximum of 79 characters:

```python
# Bad - too long
result = some_long_function_name(very_long_parameter_name1, very_long_parameter_name2, very_long_parameter_name3)

# Good - split into multiple lines
result = some_long_function_name(
    very_long_parameter_name1,
    very_long_parameter_name2,
    very_long_parameter_name3
)
```

### Imports
Group imports in the following order:
1. Standard library
2. Third-party
3. Local application/library specific

```python
# Good import organization
import os
import sys

import numpy as np
import pandas as pd

from myproject.utils import helper
from myproject.core import main

# Bad import organization
from myproject.utils import helper
import sys
import pandas as pd
from myproject.core import *  # Avoid wildcard imports
import os
```



## Tools for coding styles

### Linting

Linting is the process of analyzing your code to check for errors and enforce coding standards. 

Here are some popular linting tools for `Python`:

- [Pylint](https://pypi.org/project/pylint/): it is a static type checker and linting tool for Python.
- [Black](https://pypi.org/project/black/): it is a code formatter that automatically formats your code to adhere to a consistent style.

Please check the VS Code tutorial for python linting [here](https://code.visualstudio.com/docs/python/linting).

The most popular linting tools for `R`:

- [lintr](https://lintr.r-lib.org/articles/lintr.html): it has been integrated into the RStudio and VS Code R extensions.

### Formatting

Formatting is the process of organizing your code to make it more readable.

Please check the VS Code tutorial for python formatting [here](https://code.visualstudio.com/docs/python/formatting).
