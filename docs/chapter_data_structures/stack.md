# Stack

<a href="https://colab.research.google.com/github/junwei-lu/bst236/blob/main/bst236/codes/chapter04_data_structures.ipynb#scrollTo=6d719b8a05f8" target="_blank" style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; background: linear-gradient(135deg, #2e7d32 0%, #66bb6a 100%); color: white; border-radius: 6px; text-decoration: none; font-size: 0.85em; font-weight: 600;">▶ Try in Colab</a>

A <u>stack</u> is a linear data structure that follows the principle of Last-In-First-Out (LIFO).

We can compare a stack to a pile of plates on a table. To access the bottom plate, one must first remove the plates on top. By replacing the plates with various types of elements (such as integers, characters, objects, etc.), we obtain the data structure known as a stack.

As shown in the figure below, we refer to the top of the pile of elements as the "top of the stack" and the bottom as the "bottom of the stack." The operation of adding elements to the top of the stack is called "push," and the operation of removing the top element is called "pop."

![Stack's last-in-first-out rule](stack.assets/stack_operations.png)

## Common operations on stack

The common operations on a stack are shown in the table below. The specific method names depend on the programming language used. Here, we use `push()`, `pop()`, and `peek()` as examples.

<p align="center"> Table <id> &nbsp; Efficiency of stack operations </p>

| Method   | Description                                     | Time Complexity |
| -------- | ----------------------------------------------- | --------------- |
| `push()` | Push an element onto the stack (add to the top) | $O(1)$          |
| `pop()`  | Pop the top element from the stack              | $O(1)$          |


Typically, we can directly use the stack class built into the programming language. However, some languages may not specifically provide a stack class. In these cases, we can use the language's "array" or "linked list" as a stack and ignore operations that are not related to stack logic in the program.


```python title="stack.py"
# Initialize the stack
# Python does not have a built-in stack class, so a list can be used as a stack
stack: list[int] = []

# Push elements onto the stack
stack.append(1)
stack.append(3)
stack.append(2)
stack.append(5)
stack.append(4)

# Access the top element of the stack
peek: int = stack[-1]

# Pop an element from the stack
pop: int = stack.pop()

# Get the length of the stack
size: int = len(stack)

# Check if the stack is empty
is_empty: bool = len(stack) == 0
```
