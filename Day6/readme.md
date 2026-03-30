# 📘 Day 6 — NumPy I/O, Structured Arrays & Performance (A to Z Guide)

A complete, beginner-friendly deep dive into reading/writing data, working with structured types, understanding memory layout, and writing high-performance NumPy code.

---

## 📋 Table of Contents

1. [Why I/O Matters](#1-why-io-matters)
2. [np.save() and np.load() — Binary Format](#2-npsave-and-npload--binary-format)
3. [np.savez() — Save Multiple Arrays](#3-npsavez--save-multiple-arrays)
4. [np.savetxt() — Save as Text/CSV](#4-npsavetxt--save-as-textcsv)
5. [np.loadtxt() — Load Text/CSV Files](#5-nploadtxt--load-textcsv-files)
6. [np.genfromtxt() — Robust CSV Loading](#6-npgenfromtxt--robust-csv-loading)
7. [np.fromstring() and np.frombuffer()](#7-npfromstring-and-npfrombuffer)
8. [What are Structured Arrays?](#8-what-are-structured-arrays)
9. [Creating Structured Arrays](#9-creating-structured-arrays)
10. [Accessing & Modifying Structured Arrays](#10-accessing--modifying-structured-arrays)
11. [Nested Structured Arrays](#11-nested-structured-arrays)
12. [Record Arrays — np.recarray](#12-record-arrays--nprecarray)
13. [dtype — Data Types Deep Dive](#13-dtype--data-types-deep-dive)
14. [Views vs Copies — Memory Safety](#14-views-vs-copies--memory-safety)
15. [Memory Layout — C-order vs F-order](#15-memory-layout--c-order-vs-f-order)
16. [Strides — How Arrays Are Stored](#16-strides--how-arrays-are-stored)
17. [np.ascontiguousarray() and np.asfortranarray()](#17-npascontiguousarray-and-npasfortranarray)
18. [Memory Optimization Techniques](#18-memory-optimization-techniques)
19. [Performance Best Practices](#19-performance-best-practices)
20. [Real-World Example — CSV Pipeline](#20-real-world-example--csv-pipeline)
21. [Practice Exercises](#21-practice-exercises)
22. [Cheat Sheet](#22-cheat-sheet)

---

## 1. Why I/O Matters

In real-world projects, your data lives in **files** — CSV files, binary files, databases, or custom formats. NumPy provides tools to:

- **Save** computed arrays to disk so you don't recompute them every time
- **Load** data from CSV files, text files, or binary files
- **Share** data between programs and collaborators

Understanding I/O and memory is what separates a casual NumPy user from someone who writes **production-quality** numerical code.

---

## 2. np.save() and np.load() — Binary Format

`np.save()` saves a **single array** to a `.npy` binary file. This is the **fastest and most reliable** way to persist NumPy arrays.

### Saving

```python
import numpy as np

a = np.array([[1.5, 2.3, 3.7],
              [4.1, 5.9, 6.2]])

np.save('my_array.npy', a)
# Creates file: my_array.npy
# NumPy adds .npy extension automatically if missing
```

### Loading

```python
b = np.load('my_array.npy')
print(b)
# [[1.5 2.3 3.7]
#  [4.1 5.9 6.2]]

print(type(b))    # <class 'numpy.ndarray'>
print(b.dtype)    # float64
print(b.shape)    # (2, 3)
```

### Why use .npy instead of CSV?

| Feature | `.npy` | `.csv` |
|---------|--------|--------|
| Speed | Very fast | Slow |
| File size | Small (binary) | Large (text) |
| Dtype preserved | ✅ Yes | ❌ No (always re-inferred) |
| Shape preserved | ✅ Yes | ❌ No |
| Human readable | ❌ No | ✅ Yes |
| Interoperable | NumPy only | Any software |

Use `.npy` when you only need to share data **between Python/NumPy sessions**.
Use `.csv` when you need **human-readable** or cross-software compatibility.

---

## 3. np.savez() — Save Multiple Arrays

`np.savez()` saves **multiple arrays** into a single `.npz` file (a zip archive of `.npy` files).

### Saving

```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([10., 20., 30.])
labels = np.array(['a', 'b', 'c', 'd', 'e'])

# Save multiple arrays with keyword names
np.savez('dataset.npz', features=x, targets=y, names=labels)
```

### Loading

```python
data = np.load('dataset.npz')

print(data.files)            # ['features', 'targets', 'names']
print(data['features'])      # [1 2 3 4 5]
print(data['targets'])       # [10. 20. 30.]
print(data['names'])         # ['a' 'b' 'c' 'd' 'e']
```

### np.savez_compressed() — Smaller File Size

```python
# Compress the .npz file (slower to save, smaller file)
np.savez_compressed('dataset_compressed.npz', features=x, targets=y)
```

### Real-World Use — Save train/test split

```python
rng = np.random.default_rng(42)

X = rng.normal(0, 1, (1000, 10))   # 1000 samples, 10 features
y = rng.integers(0, 3, 1000)       # labels

# Split into 80/20 train/test
split = 800
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Save everything in one file
np.savez('train_test_split.npz',
         X_train=X_train, X_test=X_test,
         y_train=y_train, y_test=y_test)

# Later — reload and use immediately
data = np.load('train_test_split.npz')
X_train = data['X_train']
y_train = data['y_train']
print(f"Training set: {X_train.shape}, Labels: {y_train.shape}")
```

---

## 4. np.savetxt() — Save as Text/CSV

`np.savetxt()` saves an array as a **human-readable text file**, including CSV format.

### Basic Usage

```python
a = np.array([[1.5, 2.3, 3.7],
              [4.1, 5.9, 6.2],
              [7.8, 8.4, 9.1]])

# Save as CSV with comma delimiter
np.savetxt('data.csv', a, delimiter=',')
```

### With formatting options

```python
np.savetxt(
    'data.csv',
    a,
    delimiter=',',
    fmt='%.2f',           # 2 decimal places
    header='col1,col2,col3',   # header row
    comments='',          # disable '#' prefix on header
    footer='end of data'  # optional footer
)
```

### The resulting file looks like:

```
col1,col2,col3
1.50,2.30,3.70
4.10,5.90,6.20
7.80,8.40,9.10
end of data
```

### Format Codes

| Code | Meaning | Example |
|------|---------|---------|
| `%d` | Integer | `1` |
| `%f` | Float | `3.141593` |
| `%.2f` | Float, 2 decimal places | `3.14` |
| `%e` | Scientific notation | `3.14e+00` |
| `%s` | String | `hello` |
| `%10.4f` | Width 10, 4 decimals | `    3.1416` |

---

## 5. np.loadtxt() — Load Text/CSV Files

`np.loadtxt()` loads data from a **text or CSV file** into a NumPy array.

### Basic Loading

```python
# Load a CSV file
a = np.loadtxt('data.csv', delimiter=',')
print(a)
print(a.shape)
print(a.dtype)
```

### Common Parameters

```python
a = np.loadtxt(
    'data.csv',
    delimiter=',',       # column separator
    skiprows=1,          # skip header row
    usecols=(0, 1, 2),   # only load columns 0, 1, 2
    dtype=float,         # data type
    max_rows=100         # load only first 100 rows
)
```

### Loading specific columns and rows

```python
# Load only columns 1 and 3 (0-indexed), skip first row
a = np.loadtxt('data.csv', delimiter=',', skiprows=1, usecols=(1, 3))
```

### Loading with mixed types (use dtype=str first)

```python
# If your CSV has mixed types, load as strings first
raw = np.loadtxt('data.csv', delimiter=',', dtype=str, skiprows=1)
names  = raw[:, 0]
values = raw[:, 1:].astype(float)
```

### ⚠️ Limitation of loadtxt()

`loadtxt()` **fails if there are missing values (NaN)**. Use `genfromtxt()` instead for messy data.

---

## 6. np.genfromtxt() — Robust CSV Loading

`np.genfromtxt()` is a **more powerful and flexible** loader that handles:
- Missing values
- Mixed data types
- Comments
- Custom converters

```python
# data_with_missing.csv:
# name,age,score
# Alice,25,88.5
# Bob,,92.0
# Carol,30,
# Dave,28,75.5

data = np.genfromtxt(
    'data_with_missing.csv',
    delimiter=',',
    names=True,        # first row as field names
    dtype=None,        # auto-detect types
    encoding='utf-8',  # handle text encoding
    filling_values=0   # replace missing values with 0
)

print(data['name'])    # ['Alice' 'Bob' 'Carol' 'Dave']
print(data['age'])     # [25.  0. 30. 28.]   ← missing replaced with 0
print(data['score'])   # [88.5 92.   0. 75.5]
```

### Handling missing values as NaN

```python
data = np.genfromtxt(
    'data_with_missing.csv',
    delimiter=',',
    skip_header=1,
    filling_values=np.nan   # missing → NaN
)

# Clean: remove rows with any NaN
mask    = ~np.isnan(data).any(axis=1)
cleaned = data[mask]
print(f"Rows before: {len(data)}, after cleaning: {len(cleaned)}")
```

### genfromtxt() vs loadtxt()

| Feature | `loadtxt()` | `genfromtxt()` |
|---------|-------------|----------------|
| Speed | Faster | Slightly slower |
| Missing values | ❌ Crashes | ✅ Handles gracefully |
| Named columns | ❌ No | ✅ Yes (`names=True`) |
| Auto dtype | ❌ No | ✅ Yes (`dtype=None`) |
| Comments | ✅ Yes | ✅ Yes |

---

## 7. np.fromstring() and np.frombuffer()

### np.fromstring() — Create Array from a String

```python
# Parse numbers from a string
s = "1.5 2.3 3.7 4.1 5.9"
a = np.fromstring(s, dtype=float, sep=' ')
print(a)   # [1.5 2.3 3.7 4.1 5.9]

# With comma separator
s2 = "10,20,30,40"
a2 = np.fromstring(s2, dtype=int, sep=',')
print(a2)  # [10 20 30 40]
```

### np.frombuffer() — Create Array from Raw Bytes

```python
import struct

# Create raw bytes representing 4 floats
raw_bytes = struct.pack('4f', 1.0, 2.0, 3.0, 4.0)

a = np.frombuffer(raw_bytes, dtype=np.float32)
print(a)   # [1. 2. 3. 4.]
```

### np.fromiter() — Create Array from Iterator

```python
# Create array from a generator (memory efficient for large sequences)
gen = (x**2 for x in range(10))
a   = np.fromiter(gen, dtype=int)
print(a)   # [ 0  1  4  9 16 25 36 49 64 81]
```

---

## 8. What are Structured Arrays?

A **structured array** is a NumPy array where each element can have **multiple fields of different types** — like a database table or a C struct.

Without structured arrays, if you have a dataset with names (strings), ages (integers), and scores (floats), you'd need **separate arrays**:

```python
# Without structured arrays — messy
names  = np.array(['Alice', 'Bob', 'Carol'])
ages   = np.array([25, 30, 22])
scores = np.array([88.5, 92.0, 75.3])
```

With structured arrays, everything lives **in one array**:

```python
# With structured arrays — clean
data = np.array([
    ('Alice', 25, 88.5),
    ('Bob',   30, 92.0),
    ('Carol', 22, 75.3)
], dtype=[('name', 'U10'), ('age', 'i4'), ('score', 'f8')])

print(data['name'])    # ['Alice' 'Bob' 'Carol']
print(data['age'])     # [25 30 22]
print(data['score'])   # [88.5 92.  75.3]
```

---

## 9. Creating Structured Arrays

### Method 1 — Using a list of tuples for dtype

```python
# Define the structure (dtype)
dt = np.dtype([
    ('name',  'U20'),    # Unicode string, max 20 chars
    ('age',   'i4'),     # 32-bit integer
    ('height','f4'),     # 32-bit float
    ('active','bool')    # boolean
])

# Create array with this dtype
people = np.array([
    ('Alice', 28, 5.6, True),
    ('Bob',   35, 6.1, False),
    ('Carol', 22, 5.4, True),
    ('Dave',  45, 5.9, True)
], dtype=dt)

print(people)
```

### Method 2 — Using np.zeros() with structured dtype

```python
# Create an empty structured array and fill it
dt = np.dtype([('x', 'f8'), ('y', 'f8'), ('label', 'U5')])
points = np.zeros(5, dtype=dt)

points['x'] = [1.0, 2.0, 3.0, 4.0, 5.0]
points['y'] = [2.0, 4.0, 6.0, 8.0, 10.0]
points['label'] = ['A', 'B', 'C', 'D', 'E']

print(points)
```

### Method 3 — Using dictionary dtype

```python
dt = np.dtype({'names': ['id', 'value', 'flag'],
               'formats': ['i4', 'f8', 'bool']})

data = np.zeros(3, dtype=dt)
print(data.dtype)
```

### Common dtype type codes

| Code | Type | Size | Example |
|------|------|------|---------|
| `'i1'` | int8 | 1 byte | `-128` to `127` |
| `'i2'` | int16 | 2 bytes | small integers |
| `'i4'` | int32 | 4 bytes | standard integers |
| `'i8'` | int64 | 8 bytes | large integers |
| `'f4'` | float32 | 4 bytes | lower precision |
| `'f8'` | float64 | 8 bytes | standard floats |
| `'U10'` | Unicode | 10 chars | short strings |
| `'S20'` | bytes | 20 bytes | byte strings |
| `'bool'` | bool | 1 byte | True/False |
| `'M8[D]'` | datetime (days) | 8 bytes | dates |
| `'m8[s]'` | timedelta (seconds) | 8 bytes | durations |

---

## 10. Accessing & Modifying Structured Arrays

### Accessing fields

```python
people = np.array([
    ('Alice', 28, 88.5),
    ('Bob',   35, 92.0),
    ('Carol', 22, 75.3),
    ('Dave',  45, 68.1)
], dtype=[('name', 'U10'), ('age', 'i4'), ('score', 'f8')])

# Access entire field (column)
print(people['name'])    # ['Alice' 'Bob' 'Carol' 'Dave']
print(people['age'])     # [28 35 22 45]
print(people['score'])   # [88.5 92.  75.3 68.1]

# Access a single record (row)
print(people[0])         # ('Alice', 28, 88.5)
print(people[0]['name']) # 'Alice'
print(people[0]['age'])  # 28
```

### Slicing

```python
# First two records
print(people[:2])

# All records where age > 30
print(people[people['age'] > 30])
# [('Bob', 35, 92.) ('Dave', 45, 68.1)]
```

### Sorting by field

```python
# Sort by score descending
sorted_people = np.sort(people, order='score')[::-1]
print(sorted_people['name'])   # ['Bob' 'Alice' 'Carol' 'Dave']

# Sort by age, then by score
sorted_people = np.sort(people, order=['age', 'score'])
```

### Modifying fields

```python
# Update a single field
people['score'][0] = 95.0

# Conditional update
people['score'][people['age'] > 40] += 5   # bonus for seniors

# Add to all values in a field
people['age'] += 1   # everyone aged one year
```

### Multiple field access

```python
# Select multiple fields (returns a new structured array)
subset = people[['name', 'score']]
print(subset)
# [('Alice', 95.) ('Bob', 92.) ('Carol', 75.3) ('Dave', 73.1)]
```

---

## 11. Nested Structured Arrays

Fields in a structured array can themselves be **arrays or structured types**:

```python
# Each person has a name and a 2D coordinate (x, y)
dt = np.dtype([
    ('name',    'U10'),
    ('coords',  'f8', (2,)),   # sub-array: 2 floats
    ('scores',  'f4', (3,))    # sub-array: 3 floats
])

players = np.array([
    ('Alice', [1.5, 2.3], [88.0, 92.0, 85.0]),
    ('Bob',   [3.1, 4.7], [75.0, 80.0, 78.0])
], dtype=dt)

print(players['coords'])         # [[1.5 2.3] [3.1 4.7]]
print(players[0]['scores'])      # [88. 92. 85.]
print(players['scores'].mean())  # mean of all scores
```

---

## 12. Record Arrays — np.recarray

A **record array** is a structured array that lets you access fields using **attribute syntax** (dot notation) instead of string indexing.

```python
# Convert structured array to record array
people = np.array([
    ('Alice', 28, 88.5),
    ('Bob',   35, 92.0),
], dtype=[('name', 'U10'), ('age', 'i4'), ('score', 'f8')])

rec = people.view(np.recarray)

# Attribute access (dot notation)
print(rec.name)    # ['Alice' 'Bob']
print(rec.age)     # [28 35]
print(rec.score)   # [88.5 92. ]

# Index notation still works
print(rec['name']) # ['Alice' 'Bob']
```

### Create directly as recarray

```python
rec = np.rec.array([
    ('Alice', 28, 88.5),
    ('Bob',   35, 92.0),
], dtype=[('name', 'U10'), ('age', 'i4'), ('score', 'f8')])

# Chained attribute access
high_scorers = rec[rec.score > 90].name
print(high_scorers)   # ['Bob']
```

---

## 13. dtype — Data Types Deep Dive

`dtype` defines **what kind of data** each element stores and **how many bytes** it uses.

### Checking and Setting dtype

```python
a = np.array([1, 2, 3])
print(a.dtype)         # int64 (default on most systems)

b = np.array([1.5, 2.5])
print(b.dtype)         # float64

# Explicit dtype
c = np.array([1, 2, 3], dtype=np.float32)
print(c.dtype)         # float32
```

### Converting dtype — astype()

```python
a = np.array([1.7, 2.9, 3.1])
print(a.dtype)         # float64

b = a.astype(int)      # truncates decimal
print(b)               # [1 2 3]
print(b.dtype)         # int64

c = b.astype(np.float32)
print(c.dtype)         # float32
```

### ⚠️ astype() always returns a COPY

```python
a = np.array([1.5, 2.5, 3.5])
b = a.astype(int)   # new array — a is unchanged
```

### dtype object properties

```python
dt = np.dtype('float32')

print(dt.itemsize)    # 4         → bytes per element
print(dt.kind)        # 'f'       → floating point
print(dt.name)        # 'float32'
print(dt.byteorder)   # '='       → native byte order
```

### dtype kind codes

| Kind | Type | Codes |
|------|------|-------|
| `'b'` | Boolean | `bool` |
| `'i'` | Signed integer | `int8`, `int16`, `int32`, `int64` |
| `'u'` | Unsigned integer | `uint8`, `uint16`, `uint32`, `uint64` |
| `'f'` | Float | `float16`, `float32`, `float64` |
| `'c'` | Complex float | `complex64`, `complex128` |
| `'U'` | Unicode string | `str_`, `unicode_` |
| `'S'` | Byte string | `bytes_` |
| `'M'` | Datetime | `datetime64` |
| `'m'` | Timedelta | `timedelta64` |

### Choosing the Right dtype

```python
# Saves memory: use smallest dtype that fits your data
small_ints  = np.array([0, 1, 2, 3], dtype=np.uint8)   # 0-255, 1 byte each
big_ints    = np.array([1_000_000],   dtype=np.int32)   # up to ~2 billion
prices      = np.array([10.99],       dtype=np.float32) # 4 bytes vs 8 bytes

# Memory comparison
a64 = np.ones(1_000_000, dtype=np.float64)
a32 = np.ones(1_000_000, dtype=np.float32)

print(f"float64: {a64.nbytes / 1024:.0f} KB")  # 7812 KB
print(f"float32: {a32.nbytes / 1024:.0f} KB")  # 3906 KB  → half the memory!
```

---

## 14. Views vs Copies — Memory Safety

This is one of the **most common sources of bugs** in NumPy. Always know whether you have a view or a copy.

### What is a View?

A **view** is an array that **shares memory** with another array. Changing one changes the other.

```python
a = np.array([1, 2, 3, 4, 5])

view = a[1:4]      # slicing returns a VIEW
print(view)        # [2 3 4]
print(view.base is a)   # True → shares memory with a

view[0] = 99
print(a)           # [ 1 99  3  4  5]  ← ORIGINAL CHANGED!
```

### What is a Copy?

A **copy** is completely **independent**. Changing one does NOT affect the other.

```python
a = np.array([1, 2, 3, 4, 5])

copy = a[1:4].copy()   # explicit copy
copy[0] = 99
print(a)               # [1 2 3 4 5]  ← original unchanged
```

### Full Reference Table

| Operation | View or Copy? | Modifies Original? |
|-----------|---------------|-------------------|
| `a[1:3]` | View | ✅ Yes |
| `a[1:3].copy()` | Copy | ❌ No |
| `a.reshape(2,3)` | View (usually) | ✅ Yes |
| `a.ravel()` | View (usually) | ✅ Yes |
| `a.flatten()` | Copy (always) | ❌ No |
| `a[[0,1,2]]` | Copy | ❌ No |
| `a[a > 5]` | Copy | ❌ No |
| `a.T` | View | ✅ Yes |
| `np.broadcast_to(a, shape)` | View (read-only) | N/A |
| `a.astype(float)` | Copy (always) | ❌ No |
| `a.copy()` | Copy (always) | ❌ No |

### How to check

```python
a = np.array([1, 2, 3, 4, 5])
b = a[1:3]

# Method 1: check .base attribute
print(b.base is a)           # True → b is a view of a

# Method 2: check if they share memory
print(np.shares_memory(a, b))  # True

# Method 3: check .flags
print(b.flags['OWNDATA'])    # False → doesn't own its data (it's a view)
print(a.flags['OWNDATA'])    # True  → owns its data
```

### Safe Patterns

```python
# ✅ Safe: explicit copy when you intend to modify independently
def process(arr):
    arr = arr.copy()   # work on a copy, don't touch original
    arr[arr < 0] = 0
    return arr

# ✅ Safe: in-place operation on purpose (efficient, intentional)
def normalize_inplace(arr):
    arr -= arr.mean()   # modifies original — document this!
    arr /= arr.std()
```

---

## 15. Memory Layout — C-order vs F-order

How NumPy stores a 2D array in memory matters for **performance**. There are two layouts:

### C-order (Row-major) — Default in NumPy

Elements are stored **row by row** in memory:

```
Matrix:          Memory order:
[[1, 2, 3],  →  1, 2, 3, 4, 5, 6, 7, 8, 9
 [4, 5, 6],
 [7, 8, 9]]
```

### F-order (Column-major) — Fortran style

Elements are stored **column by column** in memory:

```
Matrix:          Memory order:
[[1, 2, 3],  →  1, 4, 7, 2, 5, 8, 3, 6, 9
 [4, 5, 6],
 [7, 8, 9]]
```

### Creating with specific order

```python
# C-order (default)
a_c = np.array([[1, 2, 3], [4, 5, 6]], order='C')
print(a_c.flags['C_CONTIGUOUS'])   # True
print(a_c.flags['F_CONTIGUOUS'])   # False

# F-order
a_f = np.array([[1, 2, 3], [4, 5, 6]], order='F')
print(a_f.flags['C_CONTIGUOUS'])   # False
print(a_f.flags['F_CONTIGUOUS'])   # True
```

### Why it matters for performance

```python
import time

a = np.random.rand(1000, 1000)

# Row access is fast in C-order (data is contiguous)
start = time.time()
for i in range(1000):
    _ = a[i, :]   # row access
print(f"Row access: {(time.time()-start)*1000:.2f} ms")

# Column access may be slower
start = time.time()
for i in range(1000):
    _ = a[:, i]   # column access
print(f"Col access: {(time.time()-start)*1000:.2f} ms")
```

### Rule of thumb

- Use **C-order** (default) when you access data **row by row**
- Use **F-order** when you access data **column by column** or when using Fortran-based libraries (LAPACK, BLAS)

---

## 16. Strides — How Arrays Are Stored

**Strides** tell NumPy how many **bytes to jump** to get to the next element along each dimension.

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=np.int32)

print(a.strides)    # (12, 4)
# To move 1 row down → jump 12 bytes (3 elements × 4 bytes)
# To move 1 column right → jump 4 bytes (1 element × 4 bytes)
```

### Visualizing strides

```
Memory: [1][2][3][4][5][6]
         0  4  8 12 16 20  ← byte positions (int32 = 4 bytes each)

To get a[0][0] → start at byte 0
To get a[0][1] → byte 0 + 4  = byte 4   (stride[1] = 4)
To get a[1][0] → byte 0 + 12 = byte 12  (stride[0] = 12)
```

### Strides for different dtypes

```python
a64 = np.ones((3, 4), dtype=np.float64)
a32 = np.ones((3, 4), dtype=np.float32)

print(a64.strides)   # (32, 8)   → float64 = 8 bytes
print(a32.strides)   # (16, 4)   → float32 = 4 bytes
```

### Stride tricks — Creating views with custom strides

`np.lib.stride_tricks.as_strided()` lets you create powerful views using custom strides. Use with caution!

```python
from numpy.lib.stride_tricks import as_strided

a = np.array([1, 2, 3, 4, 5, 6])

# Create overlapping windows of size 3
window_size = 3
shape   = (len(a) - window_size + 1, window_size)
strides = (a.strides[0], a.strides[0])

windows = as_strided(a, shape=shape, strides=strides)
print(windows)
# [[1 2 3]
#  [2 3 4]
#  [3 4 5]
#  [4 5 6]]
```

---

## 17. np.ascontiguousarray() and np.asfortranarray()

Sometimes arrays become **non-contiguous** in memory (e.g., after transposing). This can slow down operations that need contiguous memory.

### Check if contiguous

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
b = a.T   # transpose creates a non-C-contiguous view

print(a.flags['C_CONTIGUOUS'])   # True
print(b.flags['C_CONTIGUOUS'])   # False → b is not C-contiguous!
```

### Make contiguous

```python
# Force C-contiguous copy
b_c = np.ascontiguousarray(b)
print(b_c.flags['C_CONTIGUOUS'])  # True

# Force F-contiguous copy
b_f = np.asfortranarray(a)
print(b_f.flags['F_CONTIGUOUS'])  # True
```

### When does this matter?

Some external libraries (like BLAS, LAPACK, or Cython extensions) **require contiguous arrays**. Passing a non-contiguous array can cause errors or silently give wrong results.

```python
# Safe pattern when calling external code
def external_function_call(arr):
    arr = np.ascontiguousarray(arr)   # ensure contiguous before passing
    # ... call external library
```

---

## 18. Memory Optimization Techniques

### 1. Use the smallest dtype that fits

```python
# Check data range first, then choose dtype
data = np.array([10, 50, 200, 150, 80])

print(data.min(), data.max())   # 10, 200 → fits in uint8 (0-255)!

optimized = data.astype(np.uint8)
print(f"Original:  {data.nbytes} bytes")    # 40 bytes (int64)
print(f"Optimized: {optimized.nbytes} bytes")  # 5 bytes (uint8)
```

### 2. Use in-place operations

```python
a = np.ones((1000, 1000))

# Creates a NEW array (uses 2× memory temporarily)
a = a + 1

# In-place (no extra memory allocated)
a += 1       # same as np.add(a, 1, out=a)
a *= 2
a -= 0.5
```

### 3. Use out parameter to avoid temporaries

```python
a = np.random.rand(1000000)
b = np.random.rand(1000000)
result = np.empty_like(a)   # pre-allocate output

# Computes a+b directly into result — no temporary array
np.add(a, b, out=result)
np.sqrt(result, out=result)  # in-place sqrt
```

### 4. Delete large arrays when done

```python
large = np.random.rand(10000, 10000)
# ... use large ...
del large   # free memory immediately
```

### 5. Use np.empty() instead of np.zeros() when you will overwrite

```python
# np.zeros fills with zeros (extra work)
a = np.zeros((1000, 1000))

# np.empty just allocates memory (faster if you'll overwrite immediately)
a = np.empty((1000, 1000))
a[:] = compute_values()   # immediately overwrite with real values
```

### 6. Memory usage reporting

```python
a = np.random.rand(1000, 1000)

print(f"Elements:  {a.size:,}")
print(f"Bytes/element: {a.itemsize}")
print(f"Total bytes: {a.nbytes:,}")
print(f"Total MB: {a.nbytes / 1024**2:.2f} MB")
```

---

## 19. Performance Best Practices

### 1. Pre-allocate output arrays

```python
n = 1_000_000

# ❌ Slow: grow array one element at a time
result = np.array([])
for i in range(n):
    result = np.append(result, i * 2)

# ✅ Fast: pre-allocate, then fill
result = np.empty(n)
for i in range(n):
    result[i] = i * 2

# ✅ Fastest: use vectorized operations
result = np.arange(n) * 2
```

### 2. Avoid np.append() in loops

`np.append()` creates a **new array every time** — it is O(n) per call, making loops O(n²) overall.

```python
# ❌ Very slow
result = np.array([])
for x in range(10000):
    result = np.append(result, x)

# ✅ Use a Python list, then convert
result_list = []
for x in range(10000):
    result_list.append(x)
result = np.array(result_list)

# ✅ Or best: use np.arange/linspace/vectorize
result = np.arange(10000)
```

### 3. Use np.einsum() for complex operations

`np.einsum()` (Einstein summation) expresses complex multi-dimensional operations concisely and efficiently:

```python
a = np.random.rand(100, 50)
b = np.random.rand(50, 80)

# Matrix multiplication
c = np.einsum('ij,jk->ik', a, b)   # equivalent to a @ b

# Element-wise then sum
d = np.einsum('ij,ij->', a, a)     # equivalent to (a * a).sum()

# Batch dot product (row by row)
x = np.random.rand(100, 50)
y = np.random.rand(100, 50)
dots = np.einsum('ij,ij->i', x, y)  # 100 dot products at once
```

### 4. Use np.vectorize() wisely

```python
# ❌ np.vectorize is NOT faster than a loop — it's just syntactic sugar
vfunc = np.vectorize(lambda x: x**2 + 2*x)

# ✅ True vectorization is always faster
result = a**2 + 2*a
```

### 5. Use views instead of copies when possible

```python
# ❌ Creates a copy (uses extra memory)
row = a[0, :].copy()

# ✅ Creates a view (no extra memory, just a reference)
row = a[0, :]
# Only call .copy() if you actually need to modify independently
```

### 6. Avoid Python loops — always think in array terms

```python
data = np.random.rand(1000, 1000)

# ❌ Loop over rows
row_means = np.zeros(1000)
for i in range(1000):
    row_means[i] = data[i].mean()

# ✅ Vectorized
row_means = data.mean(axis=1)
```

### Performance Summary Table

| Situation | Slow Way | Fast Way |
|-----------|----------|----------|
| Loop over elements | `for x in arr` | Vectorized operation |
| Growing array | `np.append(arr, x)` | Pre-allocate + fill |
| Element-wise function | `np.vectorize(f)` | Built-in ufunc |
| Temporary arrays | `result = a + b + c` | `np.add(a,b,out=t); np.add(t,c,out=result)` |
| Dtype memory | `float64` by default | `float32` when precision allows |
| Row operations | Loop + `.mean()` | `.mean(axis=1)` |

---

## 20. Real-World Example — CSV Pipeline

A complete pipeline: generate data → save to CSV → load → clean → analyze → save results.

```python
import numpy as np

# ─── STEP 1: Generate synthetic employee data ─────────────────
rng = np.random.default_rng(42)
n   = 200

departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance']
dept_ids    = rng.integers(0, 5, n)

salaries    = rng.normal(70000, 15000, n).clip(30000, 150000)
experience  = rng.integers(1, 30, n)
performance = rng.normal(75, 12, n).clip(0, 100)

# Introduce 10 missing values in performance
missing_idx = rng.choice(n, 10, replace=False)
performance_with_nan = performance.astype(float)
performance_with_nan[missing_idx] = np.nan

# ─── STEP 2: Save to CSV ───────────────────────────────────────
data = np.column_stack([dept_ids, salaries, experience, performance_with_nan])
np.savetxt(
    'employees.csv',
    data,
    delimiter=',',
    fmt=['%d', '%.2f', '%d', '%.1f'],
    header='dept_id,salary,experience,performance',
    comments=''
)
print("✅ Saved employees.csv")

# ─── STEP 3: Load and clean ────────────────────────────────────
raw = np.genfromtxt('employees.csv', delimiter=',',
                    skip_header=1, filling_values=np.nan)

dept_ids_l   = raw[:, 0].astype(int)
salaries_l   = raw[:, 1]
experience_l = raw[:, 2].astype(int)
perf_l       = raw[:, 3]

# Remove rows with missing performance
valid = ~np.isnan(perf_l)
dept_ids_l   = dept_ids_l[valid]
salaries_l   = salaries_l[valid]
experience_l = experience_l[valid]
perf_l       = perf_l[valid]
print(f"✅ Loaded: {raw.shape[0]} rows, cleaned to: {valid.sum()} rows")

# ─── STEP 4: Analyze ──────────────────────────────────────────
print("\n📊 Analysis by Department")
print(f"{'Dept':<14} {'N':>4} {'Avg Salary':>12} {'Avg Exp':>9} {'Avg Perf':>10}")
print("-" * 52)

for d_id, d_name in enumerate(departments):
    mask = dept_ids_l == d_id
    if mask.sum() == 0:
        continue
    s = salaries_l[mask]
    e = experience_l[mask]
    p = perf_l[mask]
    print(f"{d_name:<14} {mask.sum():>4} "
          f"{s.mean():>12,.0f} "
          f"{e.mean():>9.1f} "
          f"{p.mean():>10.1f}")

# ─── STEP 5: Save structured results ──────────────────────────
dt = np.dtype([
    ('dept',        'U15'),
    ('avg_salary',  'f8'),
    ('avg_exp',     'f4'),
    ('avg_perf',    'f4'),
    ('headcount',   'i4')
])

results = np.zeros(len(departments), dtype=dt)
for d_id, d_name in enumerate(departments):
    mask = dept_ids_l == d_id
    results[d_id]['dept']       = d_name
    results[d_id]['avg_salary'] = salaries_l[mask].mean() if mask.sum() > 0 else 0
    results[d_id]['avg_exp']    = experience_l[mask].mean() if mask.sum() > 0 else 0
    results[d_id]['avg_perf']   = perf_l[mask].mean() if mask.sum() > 0 else 0
    results[d_id]['headcount']  = mask.sum()

np.save('dept_summary.npy', results)
print("\n✅ Saved dept_summary.npy")

# Reload and verify
loaded = np.load('dept_summary.npy', allow_pickle=True)
print("\nTop department by performance:")
top = loaded[loaded['avg_perf'].argmax()]
print(f"  {top['dept']}: avg performance = {top['avg_perf']:.1f}")
```

---

## 21. Practice Exercises

### Exercise 1 — File I/O
```python
# Q1: Create a 4×4 array of random floats, save it as 'matrix.npy',
#     reload it, and verify the values match

# Q2: Save 3 arrays (X, y, weights) to a single .npz file,
#     reload them, and print their shapes

# Q3: Save a 5×3 array to 'data.csv' with headers col1,col2,col3
#     and 3 decimal places, then reload it with loadtxt
```

<details>
<summary>Show Answers</summary>

```python
rng = np.random.default_rng(42)

# Q1
original = rng.random((4, 4))
np.save('matrix.npy', original)
loaded = np.load('matrix.npy')
print(np.allclose(original, loaded))   # True

# Q2
X = rng.random((100, 5))
y = rng.integers(0, 3, 100)
w = rng.random(5)
np.savez('arrays.npz', X=X, y=y, weights=w)
data = np.load('arrays.npz')
print(data['X'].shape, data['y'].shape, data['weights'].shape)

# Q3
arr = rng.random((5, 3))
np.savetxt('data.csv', arr, delimiter=',',
           header='col1,col2,col3', comments='', fmt='%.3f')
reloaded = np.loadtxt('data.csv', delimiter=',', skiprows=1)
print(np.allclose(arr, reloaded, atol=1e-3))   # True
```
</details>

---

### Exercise 2 — Structured Arrays
```python
# Create a structured array for a product catalog with:
# - product_id (int32)
# - name (U20 string)
# - price (float32)
# - in_stock (bool)
#
# Add 5 products, then:
# Q1: Print all products where price > 50
# Q2: Sort by price (ascending) and print names
# Q3: Calculate the total value of in-stock items
#     (sum of prices for products where in_stock=True)
```

<details>
<summary>Show Answers</summary>

```python
dt = np.dtype([('product_id','i4'),('name','U20'),('price','f4'),('in_stock','bool')])
catalog = np.array([
    (1, 'Widget A',  29.99, True),
    (2, 'Widget B',  74.50, True),
    (3, 'Gadget X',  12.00, False),
    (4, 'Gadget Y', 149.00, True),
    (5, 'Tool Z',    55.00, False),
], dtype=dt)

# Q1
print(catalog[catalog['price'] > 50]['name'])

# Q2
sorted_cat = np.sort(catalog, order='price')
print(sorted_cat['name'])

# Q3
in_stock = catalog[catalog['in_stock']]
print(f"Total in-stock value: ${in_stock['price'].sum():.2f}")
```
</details>

---

### Exercise 3 — Views vs Copies
```python
a = np.arange(20).reshape(4, 5)

# Q1: Create a view of rows 1-2, modify element [0,0] to 999.
#     Does a change?
# Q2: Create a copy of column 3, modify its first element.
#     Does a change?
# Q3: Check using np.shares_memory()
```

<details>
<summary>Show Answers</summary>

```python
# Q1
view = a[1:3]
view[0, 0] = 999
print(a[1, 0])               # 999 → yes, a changed

# Q2
a = np.arange(20).reshape(4, 5)
col_copy = a[:, 3].copy()
col_copy[0] = 888
print(a[0, 3])               # 3 → unchanged

# Q3
a = np.arange(20).reshape(4, 5)
v = a[1:3]
c = a[:, 3].copy()
print(np.shares_memory(a, v))   # True  → view
print(np.shares_memory(a, c))   # False → copy
```
</details>

---

### Exercise 4 — Memory & dtype
```python
# Q1: Create a 1M element array of float64.
#     Convert to float32. Compare memory usage.

# Q2: Create an array of integers 0-127.
#     What is the most memory-efficient unsigned dtype?
#     How much memory does it save vs int64?

# Q3: Create a 500×500 C-order array and a 500×500 F-order array.
#     Check their flags and strides.
```

<details>
<summary>Show Answers</summary>

```python
# Q1
a64 = np.random.rand(1_000_000)
a32 = a64.astype(np.float32)
print(f"float64: {a64.nbytes:,} bytes")
print(f"float32: {a32.nbytes:,} bytes → {a64.nbytes/a32.nbytes:.0f}x smaller")

# Q2
a = np.arange(128, dtype=np.int64)
b = a.astype(np.uint8)
print(f"int64:  {a.nbytes} bytes")
print(f"uint8:  {b.nbytes} bytes → {a.nbytes/b.nbytes:.0f}x savings")

# Q3
c_arr = np.ones((500, 500), order='C')
f_arr = np.ones((500, 500), order='F')
print("C-order strides:", c_arr.strides)
print("F-order strides:", f_arr.strides)
print("C contiguous:", c_arr.flags['C_CONTIGUOUS'])
print("F contiguous:", f_arr.flags['F_CONTIGUOUS'])
```
</details>

---

## 22. Cheat Sheet

```python
import numpy as np

# ── SAVING ────────────────────────────────────────────────
np.save('file.npy', arr)                   # save single array (binary)
np.savez('file.npz', a=arr1, b=arr2)       # save multiple arrays
np.savez_compressed('file.npz', a=arr1)    # compressed multi-array save
np.savetxt('file.csv', arr, delimiter=',', # save as text/CSV
           fmt='%.4f', header='c1,c2',
           comments='')

# ── LOADING ───────────────────────────────────────────────
arr  = np.load('file.npy')                 # load .npy
data = np.load('file.npz')                 # load .npz
arr  = data['a']                           # access named array
arr  = np.loadtxt('file.csv',              # load text/CSV
                  delimiter=',', skiprows=1,
                  usecols=(0,1), dtype=float)
arr  = np.genfromtxt('file.csv',           # robust CSV with missing data
                     delimiter=',', names=True,
                     dtype=None, encoding='utf-8',
                     filling_values=np.nan)

# ── STRUCTURED ARRAYS ─────────────────────────────────────
dt   = np.dtype([('name','U10'),('age','i4'),('score','f8')])
data = np.array([('Alice',25,88.5)], dtype=dt)
data['name']                               # access field
data[data['age'] > 30]                     # filter by field
np.sort(data, order='score')               # sort by field
rec  = data.view(np.recarray)              # attribute access
rec.name                                   # dot notation

# ── DTYPE ─────────────────────────────────────────────────
arr.dtype                                  # check dtype
arr.dtype.itemsize                         # bytes per element
arr.astype(np.float32)                     # convert dtype (copy!)
np.array([1,2,3], dtype=np.uint8)          # set dtype on creation

# ── VIEWS vs COPIES ───────────────────────────────────────
view = arr[1:3]                            # view (shares memory)
copy = arr[1:3].copy()                     # explicit copy
np.shares_memory(a, b)                     # check shared memory
arr.flags['OWNDATA']                       # True = owns data
arr.flags['C_CONTIGUOUS']                  # True = C-contiguous
arr.flags['F_CONTIGUOUS']                  # True = F-contiguous

# ── MEMORY LAYOUT ─────────────────────────────────────────
np.array(data, order='C')                  # C-order (row-major)
np.array(data, order='F')                  # F-order (column-major)
np.ascontiguousarray(arr)                  # make C-contiguous copy
np.asfortranarray(arr)                     # make F-contiguous copy
arr.strides                                # bytes per step in each dim

# ── MEMORY OPTIMIZATION ───────────────────────────────────
arr.nbytes                                 # total bytes used
arr.itemsize                               # bytes per element
arr.size                                   # total element count
arr += 1                                   # in-place (no extra memory)
np.add(a, b, out=result)                   # out param avoids temporaries
del arr                                    # free memory immediately
np.empty((m, n))                           # allocate without initializing
```

---

## 🔗 What's Next?

After mastering Day 6, you're ready for the final step:

➡️ **Day 7 — Capstone: Real Data Mini-Project**
Apply everything from Days 1–6 to a real-world dataset: load CSV data, clean it, compute statistics, find patterns, and visualize results with Matplotlib.

---

## 📚 Resources

- [NumPy I/O Docs](https://numpy.org/doc/stable/reference/routines.io.html)
- [NumPy Data Types](https://numpy.org/doc/stable/reference/arrays.dtypes.html)
- [NumPy Structured Arrays](https://numpy.org/doc/stable/user/basics.rec.html)
- [NumPy Memory Layout](https://numpy.org/doc/stable/reference/arrays.ndarray.html#memory-layout)
- [Practice on Google Colab](https://colab.research.google.com/)

---

*Part of the [7-Day NumPy Learning Plan](./README.md) · Day 6 of 7*
