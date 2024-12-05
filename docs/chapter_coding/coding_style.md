# Coding in style

Please check the complete Python style guide [PEP 8](https://peps.python.org/pep-0008/) or [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). You can also check the detailed [Tidyverse R style guide](https://style.tidyverse.org/).



## Tools for coding styles

However, instead of reading the whole style guide, we recommend you to use many existing tools to help you write code in a consistent style.

### Formatting

Formatting is the process of organizing your code to make it more readable.

- Python: You can check the VS Code tutorial for python formatting [here](https://code.visualstudio.com/docs/python/formatting). We recommend using VS Code extensions (e.g. [autopep8](https://marketplace.visualstudio.com/items?itemName=ms-python.autopep8)) to format your code.
- R: The R package [styler](https://styler.r-lib.org/) allows you to interactively restyle selected text, files, or entire projects. It includes an RStudio add-in, the easiest way to re-style existing code.

### Linting

Linting is the process of analyzing your code to check for errors and enforce coding standards. 

Here are some popular linting tools for `Python`:

- [Pylint](https://pypi.org/project/pylint/): it is a static type checker and linting tool for Python.
- [Black](https://pypi.org/project/black/): it is a code formatter that automatically formats your code to adhere to a consistent style.

Please check the VS Code tutorial for python linting [here](https://code.visualstudio.com/docs/python/linting).

The most popular linting tools for `R`:

- [lintr](https://lintr.r-lib.org/articles/lintr.html): it has been integrated into the RStudio and VS Code R extensions.


!!! tip "Tip:AI-assisted styling"

    You can also ask AI to help you style your code. You can document all of your code style preferences and attached the instructions when asking AI to style your code. For example, you can prompt:

    ```
    Format the following code to adhere to the PEP 8 Style Guide.
    ```





## Basic coding style principles

Despite the existence of many tools, we still list some basic coding style principles here.

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
- `_single_leading_underscore`: weak “internal use” indicator. E.g. from M import * does not import objects whose names start with an underscore.
- `single_trailing_underscore_`: used by convention to avoid conflicts with Python keyword, e.g. `class_`.
- `__double_leading_underscore`: when naming a class attribute, invokes name mangling (inside class FooBar, __boo becomes _FooBar__boo; see below).
- `__double_leading_and_trailing_underscore__`: “magic” objects or attributes that live in user-controlled namespaces. E.g. `__init__`, `__import__` or `__file__`. Never invent such names; only use them as documented.


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


