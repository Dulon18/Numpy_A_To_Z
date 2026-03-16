# 📘 NumPy — Day 1: Basics & Array Creation (A to Z)

> **Goal:** Understand what NumPy is, why it exists, and create & inspect arrays with full confidence.

---

## 📑 Table of Contents

1. [What is NumPy?](#1-what-is-numpy)
2. [Why NumPy Instead of Python Lists?](#2-why-numpy-instead-of-python-lists)
3. [Installing & Importing NumPy](#3-installing--importing-numpy)
4. [The ndarray — Core Concept](#4-the-ndarray--core-concept)
5. [Creating Arrays — From Python Data](#5-creating-arrays--from-python-data)
6. [Creating Arrays — Built-in Functions](#6-creating-arrays--built-in-functions)
7. [Data Types (dtype)](#7-data-types-dtype)
8. [Array Attributes](#8-array-attributes)
9. [Basic Array Operations (Preview)](#9-basic-array-operations-preview)
10. [Special Array Values](#10-special-array-values)
11. [Copying Arrays — Views vs Copies](#11-copying-arrays--views-vs-copies)
12. [Practice Exercises](#12-practice-exercises)
13. [Day 1 Checklist](#-day-1-checklist)

---

## 📌 1. What is NumPy?

**NumPy** (Numerical Python) is the foundational library for scientific computing in Python. It provides:

- A powerful **N-dimensional array** object (`ndarray`)
- Fast **mathematical operations** (without writing loops)
- Tools for **linear algebra**, **random numbers**, **Fourier transforms**, and more
- The backbone of popular libraries like **Pandas**, **TensorFlow**, **scikit-learn**, and **OpenCV**

NumPy is written in C under the hood, which makes it dramatically faster than plain Python for numerical tasks.

---

## 📌 2. Why NumPy Instead of Python Lists?

| Feature | Python List | NumPy Array |
|--------|-------------|-------------|
| Speed | Slow (interpreted) | Fast (compiled C) |
| Memory | More (stores objects) | Less (typed buffer) |
| Element type | Mixed types allowed | Single type only |
| Math operations | Manual loops needed | Vectorized (no loops) |
| Multi-dimensional | Nested lists only | Native N-D support |

**Speed comparison example:**

```python
import numpy as np
import time

py_list = list(range(1_000_000))
np_arr  = np.arange(1_000_000)

# Python list sum
t1 = time.time(); sum(py_list); t2 = time.time()
print(f"List:  {t2-t1:.4f}s")

# NumPy sum
t1 = time.time(); np_arr.sum(); t2 = time.time()
print(f"NumPy: {t2-t1:.4f}s")   # ~10–100x faster
```

---

## 📌 3. Installing & Importing NumPy

**Install:**
```bash
pip install numpy
```

**Import:**
```python
import numpy as np        # 'np' is the universal alias — always use this
print(np.__version__)     # check your version
```

---

## 📌 4. The ndarray — Core Concept

A NumPy **ndarray** (N-dimensional array) is:
- A grid of values, all of the **same data type**
- Indexed by a **tuple of non-negative integers**
- Characterized by its **shape** (dimensions) and **dtype** (element type)

```
1D array → [1, 2, 3, 4]                     → shape: (4,)
2D array → [[1, 2], [3, 4]]                 → shape: (2, 2)
3D array → [[[1,2],[3,4]], [[5,6],[7,8]]]   → shape: (2, 2, 2)
```

Think of it like this:
- **1D** = a single row of numbers (vector)
- **2D** = a table/matrix (rows and columns)
- **3D** = a stack of tables (like an RGB image)

---

## 📌 5. Creating Arrays — From Python Data

### From a list (1D array):
```python
a = np.array([10, 20, 30, 40])
print(a)        # [10 20 30 40]
print(type(a))  # <class 'numpy.ndarray'>
```

### From a nested list (2D matrix):
```python
b = np.array([[1, 2, 3],
              [4, 5, 6]])
print(b)
# [[1 2 3]
#  [4 5 6]]
print(b.shape)  # (2, 3)  → 2 rows, 3 columns
```

### From a nested list (3D tensor):
```python
c = np.array([[[1, 2], [3, 4]],
              [[5, 6], [7, 8]]])
print(c.shape)  # (2, 2, 2)
```

> ⚠️ **Important:** All rows must have the same length. Unequal rows create an object array — not what you want for math.

---

## 📌 6. Creating Arrays — Built-in Functions

These are the most commonly used array creation functions:

### `np.zeros(shape)` — filled with 0.0
```python
a = np.zeros((3, 4))     # 3 rows, 4 columns
print(a)
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]
```

### `np.ones(shape)` — filled with 1.0
```python
b = np.ones((2, 3))
print(b)
# [[1. 1. 1.]
#  [1. 1. 1.]]
```

### `np.full(shape, fill_value)` — filled with any value
```python
c = np.full((2, 2), 99)
print(c)
# [[99 99]
#  [99 99]]
```

### `np.eye(n)` — identity matrix (1s on diagonal, 0s elsewhere)
```python
d = np.eye(3)
print(d)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

### `np.arange(start, stop, step)` — like Python `range()`
```python
e = np.arange(0, 10, 2)
print(e)   # [0 2 4 6 8]

f = np.arange(5)
print(f)   # [0 1 2 3 4]   ← start defaults to 0, step to 1
```

> ⚠️ `stop` is **exclusive** — `np.arange(0, 10)` gives `[0..9]`, not `[0..10]`.

### `np.linspace(start, stop, num)` — `num` evenly spaced values
```python
g = np.linspace(0, 1, 5)
print(g)   # [0.   0.25 0.5  0.75 1.  ]

h = np.linspace(0, 100, 6)
print(h)   # [  0.  20.  40.  60.  80. 100.]
```

> Unlike `arange`, `linspace` **includes** the stop value. Use it when you need a fixed count of points (e.g. plotting).

### `np.empty(shape)` — uninitialized array (fastest, garbage values)
```python
i = np.empty((2, 3))   # values are whatever is in memory — don't use without filling
```

### Quick Reference Table

| Function | Output | Default dtype |
|----------|--------|---------------|
| `np.zeros((r, c))` | All zeros | `float64` |
| `np.ones((r, c))` | All ones | `float64` |
| `np.full((r, c), v)` | All `v` | inferred from `v` |
| `np.eye(n)` | Identity matrix | `float64` |
| `np.arange(a, b, s)` | Range of values | `int64` or `float64` |
| `np.linspace(a, b, n)` | `n` evenly spaced floats | `float64` |
| `np.empty((r, c))` | Uninitialized | `float64` |

---

## 📌 7. Data Types (`dtype`)

Every NumPy array stores all its elements as the **same data type**, called `dtype`. NumPy infers it automatically, but you can also set it explicitly.

### Auto-detection:
```python
a = np.array([1, 2, 3])
print(a.dtype)          # int64

b = np.array([1.0, 2.5, 3.7])
print(b.dtype)          # float64

c = np.array([True, False, True])
print(c.dtype)          # bool

d = np.array(['hello', 'world'])
print(d.dtype)          # <U5  (Unicode string, max 5 chars)
```

### Set dtype explicitly:
```python
a = np.array([1, 2, 3], dtype=np.float32)
b = np.array([1, 2, 3], dtype=np.int8)
c = np.zeros((3,), dtype=np.complex128)
```

### Convert dtype with `astype()`:
```python
a = np.array([1, 2, 3])          # int64
b = a.astype(np.float64)          # → float64
c = b.astype(np.int32)            # → int32  (decimals get truncated)

print(a.dtype, b.dtype, c.dtype)
# int64  float64  int32
```

### Common dtypes reference:

| dtype | Bits | Range / Notes |
|-------|------|---------------|
| `int8` | 8 | -128 to 127 |
| `int16` | 16 | -32,768 to 32,767 |
| `int32` | 32 | ~±2 billion |
| `int64` | 64 | ~±9.2 × 10¹⁸ **(default integer)** |
| `uint8` | 8 | 0 to 255 — great for images |
| `float32` | 32 | ~7 decimal digits of precision |
| `float64` | 64 | ~15 decimal digits **(default float)** |
| `bool` | 8 | `True` / `False` |
| `complex64` | 64 | real + imaginary (float32 each) |
| `complex128` | 128 | real + imaginary (float64 each) |

> 💡 Use `float32` and `int32` in deep learning to save memory and speed up GPU computation.

---

## 📌 8. Array Attributes

After creating an array, use these attributes to inspect it:

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.ndim)      # 2          — number of dimensions
print(a.shape)     # (2, 3)     — (rows, columns)
print(a.size)      # 6          — total number of elements
print(a.dtype)     # int64      — element data type
print(a.nbytes)    # 48         — bytes in memory  (6 elements × 8 bytes each)
print(a.itemsize)  # 8          — bytes per single element
```

| Attribute | Return type | Description |
|-----------|-------------|-------------|
| `ndim` | `int` | Number of axes (dimensions) |
| `shape` | `tuple` | Size along each axis |
| `size` | `int` | Total count of elements |
| `dtype` | `dtype` | Data type of each element |
| `nbytes` | `int` | Total memory used in bytes |
| `itemsize` | `int` | Memory per element in bytes |

### Understanding `shape` deeply:

```python
a = np.array([1, 2, 3])              # shape (3,)       — 1D: 3 elements
b = np.array([[1, 2], [3, 4]])       # shape (2, 2)     — 2D: 2 rows, 2 cols
c = np.zeros((4, 3, 2))              # shape (4, 3, 2)  — 3D: 4 blocks, 3 rows, 2 cols
```

> `shape` is a tuple: `(rows, columns)` for 2D, `(depth, rows, columns)` for 3D.

---

## 📌 9. Basic Array Operations (Preview)

Even on Day 1, you can do math on arrays without any loops:

### Scalar operations (apply to every element):
```python
a = np.array([10, 20, 30, 40])

print(a + 5)       # [15 25 35 45]
print(a - 3)       # [ 7 17 27 37]
print(a * 2)       # [20 40 60 80]
print(a / 10)      # [1. 2. 3. 4.]
print(a ** 2)      # [ 100  400  900 1600]
print(a % 3)       # [1 2 0 1]
print(a > 20)      # [False False  True  True]
```

### Two arrays (element-wise):
```python
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print(x + y)    # [5 7 9]
print(x * y)    # [ 4 10 18]
print(x - y)    # [-3 -3 -3]
print(x / y)    # [0.25 0.4  0.5]
```

> These operations are **vectorized** — they run at C speed, not Python loop speed. Day 3 covers this in full depth.

---

## 📌 10. Special Array Values

NumPy has built-in constants for special numeric values:

```python
print(np.nan)    # nan    — Not a Number (used for missing data)
print(np.inf)    # inf    — Positive infinity
print(-np.inf)   # -inf   — Negative infinity
print(np.pi)     # 3.141592653589793
print(np.e)      # 2.718281828459045
```

### Detecting NaN values:
```python
a = np.array([1.0, np.nan, 3.0, np.nan])
print(np.isnan(a))        # [False  True False  True]
print(np.isnan(a).sum())  # 2  — count of NaN values
```

### Detecting infinite values:
```python
b = np.array([1.0, np.inf, -np.inf, 4.0])
print(np.isinf(b))    # [False  True  True False]
```

> In real datasets, `np.nan` is the standard placeholder for missing values — you'll use it heavily from Day 6 onwards.

---

## 📌 11. Copying Arrays — Views vs Copies

This is one of the most common sources of bugs in NumPy.

### ⚠️ Assignment does NOT copy:
```python
a = np.array([1, 2, 3])

b = a              # b points to the SAME data as a
b[0] = 99
print(a)           # [99  2  3]  ← a was changed too!
print(b)           # [99  2  3]
```

### ✅ Use `.copy()` for a true independent copy:
```python
a = np.array([1, 2, 3])

c = a.copy()       # c is a completely separate array
c[0] = 999
print(a)           # [1 2 3]   ← a is unchanged
print(c)           # [999 2 3]
```

### Check if two arrays share memory:
```python
a = np.array([1, 2, 3])
b = a
c = a.copy()

print(np.shares_memory(a, b))   # True  — same data
print(np.shares_memory(a, c))   # False — independent
```

> **Rule of thumb:** Whenever you want to modify an array without affecting the original, always use `.copy()`.

---

## 📌 12. Practice Exercises

Work through all 10 exercises in a Jupyter Notebook or Google Colab:

```python
import numpy as np

# Exercise 1: Create a 1D array of integers from 1 to 15
a = np.arange(1, 16)
print("Ex1:", a)

# Exercise 2: Create a 5×5 matrix of all zeros
b = np.zeros((5, 5))
print("Ex2:\n", b)

# Exercise 3: Create a 5×5 identity matrix
c = np.eye(5)
print("Ex3:\n", c)

# Exercise 4: Create 10 evenly spaced values from 0 to 50
d = np.linspace(0, 50, 10)
print("Ex4:", d)

# Exercise 5: Create a 3×4 matrix filled with the value 7
e = np.full((3, 4), 7)
print("Ex5:\n", e)

# Exercise 6: Inspect all attributes of array `e`
print(f"ndim={e.ndim}, shape={e.shape}, size={e.size}, dtype={e.dtype}, nbytes={e.nbytes}")

# Exercise 7: Create an int array and convert it to float32
f = np.array([1, 2, 3, 4, 5])
f_float = f.astype(np.float32)
print("Ex7:", f_float, "| dtype:", f_float.dtype)

# Exercise 8: Add 100 to every element of `a` without a loop
print("Ex8:", a + 100)

# Exercise 9: Create a copy of `a`, modify it, and confirm `a` is unchanged
g = a.copy()
g[0] = 999
print("Ex9 — original a[0]:", a[0], "| modified copy g[0]:", g[0])

# Exercise 10: From array `d`, extract only values greater than 25
print("Ex10:", d[d > 25])
```

**Expected outputs:**
```
Ex1:  [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
Ex4:  [ 0.          5.55555556 11.11111111 16.66666667 22.22222222
       27.77777778 33.33333333 38.88888889 44.44444444 50.        ]
Ex6:  ndim=2, shape=(3, 4), size=12, dtype=int64, nbytes=96
Ex7:  [1. 2. 3. 4. 5.] | dtype: float32
Ex10: [27.77777778 33.33333333 38.88888889 44.44444444 50.        ]
```

---

## ✅ Day 1 Checklist

Before moving to Day 2, make sure you can answer **yes** to all of these:

- [ ] I know what NumPy is and why it's faster than Python lists
- [ ] I can import NumPy using `import numpy as np`
- [ ] I can create 1D, 2D, and 3D arrays from Python lists
- [ ] I can use `np.zeros`, `np.ones`, `np.full`, `np.eye`, `np.arange`, `np.linspace`
- [ ] I understand what `dtype` is and can set or convert it with `astype()`
- [ ] I can read `ndim`, `shape`, `size`, `dtype`, `nbytes`, `itemsize` from any array
- [ ] I understand the difference between a **view** and a **copy** and use `.copy()` safely
- [ ] I can perform basic arithmetic on arrays without writing any loops
- [ ] I know what `np.nan` and `np.inf` are and how to detect them

---

## 📚 Useful Resources

- [NumPy Official Docs — Array creation](https://numpy.org/doc/stable/user/basics.creation.html)
- [NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [Google Colab](https://colab.research.google.com/) — free browser-based notebook, no setup needed

---

*Next up → **Day 2: Indexing, Slicing & Reshaping** 🚀*
