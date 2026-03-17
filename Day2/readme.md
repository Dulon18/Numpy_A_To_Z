# 📘 Day 2 — NumPy Indexing, Slicing & Reshaping (A to Z Guide)

A complete, beginner-friendly deep dive into extracting, filtering, and rearranging NumPy arrays.

---

## 📋 Table of Contents

1. [What is Indexing?](#1-what-is-indexing)
2. [1D Array Indexing](#2-1d-array-indexing)
3. [2D Array Indexing](#3-2d-array-indexing)
4. [3D Array Indexing](#4-3d-array-indexing)
5. [Slicing](#5-slicing)
6. [Slicing 2D Arrays](#6-slicing-2d-arrays)
7. [Boolean Indexing](#7-boolean-indexing)
8. [Fancy Indexing](#8-fancy-indexing)
9. [reshape()](#9-reshape)
10. [flatten() vs ravel()](#10-flatten-vs-ravel)
11. [transpose()](#11-transpose)
12. [Views vs Copies](#12-views-vs-copies)
13. [Practice Exercises](#13-practice-exercises)
14. [Cheat Sheet](#14-cheat-sheet)

---

## 1. What is Indexing?

Indexing means **accessing a specific element** from an array using its position number (called an **index**).

- NumPy arrays are **zero-indexed** — the first element is at position `0`, not `1`.
- Negative indices count from the **end** of the array.

```
Array:   [10,  20,  30,  40,  50]
Index:     0    1    2    3    4
Negative: -5   -4   -3   -2   -1
```

---

## 2. 1D Array Indexing

A 1D array is like a simple list.

```python
import numpy as np

a = np.array([10, 20, 30, 40, 50])

print(a[0])    # 10  → first element
print(a[2])    # 30  → third element
print(a[-1])   # 50  → last element
print(a[-2])   # 40  → second from last
```

### ✅ Key Rule
| Expression | Meaning           |
|------------|-------------------|
| `a[0]`     | First element     |
| `a[-1]`    | Last element      |
| `a[n]`     | Element at index n |

---

## 3. 2D Array Indexing

A 2D array is like a **table with rows and columns**.

```python
a = np.array([[1,  2,  3],
              [4,  5,  6],
              [7,  8,  9]])

print(a[0, 0])   # 1  → row 0, col 0
print(a[1, 2])   # 6  → row 1, col 2
print(a[2, 1])   # 8  → row 2, col 1
print(a[-1, -1]) # 9  → last row, last col
```

### How to Think About It

```
          col0  col1  col2
row 0  →  [ 1,   2,   3 ]
row 1  →  [ 4,   5,   6 ]
row 2  →  [ 7,   8,   9 ]
```

`a[row, column]` — always row first, then column.

---

## 4. 3D Array Indexing

A 3D array can be thought of as **multiple 2D tables stacked together** (like pages of a book).

```python
a = np.array([[[1,  2],  [3,  4]],
              [[5,  6],  [7,  8]]])

# Shape is (2, 2, 2) → 2 blocks, 2 rows, 2 cols

print(a[0, 0, 0])  # 1  → block 0, row 0, col 0
print(a[1, 0, 1])  # 6  → block 1, row 0, col 1
print(a[1, 1, 1])  # 8  → block 1, row 1, col 1
```

### Syntax: `a[block, row, col]`

---

## 5. Slicing

Slicing means **extracting a range of elements** from an array — like cutting out a piece.

### Syntax: `a[start : stop : step]`

| Part    | Meaning                              | Default |
|---------|--------------------------------------|---------|
| `start` | Index to begin from (inclusive)      | `0`     |
| `stop`  | Index to end at (exclusive)          | end     |
| `step`  | How many positions to jump each time | `1`     |

```python
a = np.array([10, 20, 30, 40, 50, 60, 70])

print(a[1:4])     # [20 30 40]   → index 1 to 3
print(a[:3])      # [10 20 30]   → from start to index 2
print(a[4:])      # [50 60 70]   → index 4 to end
print(a[::2])     # [10 30 50 70] → every 2nd element
print(a[1::2])    # [20 40 60]   → every 2nd, starting at index 1
print(a[::-1])    # [70 60 50 40 30 20 10] → reversed!
```

### ⚠️ Stop is EXCLUSIVE

`a[1:4]` gives elements at index 1, 2, 3 — NOT index 4.

---

## 6. Slicing 2D Arrays

You can slice **both rows and columns** at the same time.

```python
a = np.array([[ 1,  2,  3,  4],
              [ 5,  6,  7,  8],
              [ 9, 10, 11, 12],
              [13, 14, 15, 16]])
```

### Selecting rows

```python
print(a[0, :])    # [ 1  2  3  4]  → entire first row
print(a[2, :])    # [ 9 10 11 12]  → entire third row
print(a[:, 1])    # [ 2  6 10 14]  → entire second column
```

### Selecting a sub-matrix (block)

```python
print(a[0:2, 0:2])
# [[ 1  2]
#  [ 5  6]]

print(a[1:3, 2:4])
# [[ 7  8]
#  [11 12]]
```

### Selecting every other row/column

```python
print(a[::2, ::2])
# [[ 1  3]
#  [ 9 11]]
```

### Visual Guide

```
         col0  col1  col2  col3
row 0  → [  1,   2,   3,   4 ]
row 1  → [  5,   6,   7,   8 ]   a[1:3, 1:3] → [[6,7],[10,11]]
row 2  → [  9,  10,  11,  12 ]
row 3  → [ 13,  14,  15,  16 ]
```

---

## 7. Boolean Indexing

Boolean indexing lets you **filter elements** using a True/False condition. NumPy evaluates the condition for every element and returns only the ones where the result is `True`.

```python
a = np.array([10, 25, 3, 47, 8, 60, 15])

# Step 1: Create a boolean mask
mask = a > 20
print(mask)    # [False  True False  True False  True False]

# Step 2: Apply the mask to filter
print(a[mask]) # [25 47 60]

# One-liner (most common way)
print(a[a > 20])  # [25 47 60]
```

### More Examples

```python
# Elements equal to a value
print(a[a == 10])    # [10]

# Multiple conditions using & (and) / | (or)
print(a[(a > 10) & (a < 50)])   # [25 47 15]
print(a[(a < 5) | (a > 50)])    # [ 3 60]

# Modify elements matching a condition
a[a < 10] = 0
print(a)   # [10 25  0 47  0 60 15]
```

### ⚠️ Important
Use `&` instead of `and`, and `|` instead of `or` when combining conditions in NumPy.

---

## 8. Fancy Indexing

Fancy indexing means **passing a list of indices** to select multiple specific elements at once.

```python
a = np.array([100, 200, 300, 400, 500])

# Select elements at index 0, 2, and 4
print(a[[0, 2, 4]])   # [100 300 500]

# Select in any order
print(a[[4, 1, 3]])   # [500 200 400]

# Repeat an index
print(a[[0, 0, 2]])   # [100 100 300]
```

### Fancy Indexing on 2D Arrays

```python
a = np.array([[10, 20, 30],
              [40, 50, 60],
              [70, 80, 90]])

# Select rows 0 and 2
print(a[[0, 2]])
# [[10 20 30]
#  [70 80 90]]

# Select specific (row, col) pairs
rows = [0, 1, 2]
cols = [2, 0, 1]
print(a[rows, cols])   # [30 40 80]
# → a[0,2]=30, a[1,0]=40, a[2,1]=80
```

### Difference: Slicing vs Fancy Indexing

| Feature | Slicing | Fancy Indexing |
|---------|---------|----------------|
| Syntax | `a[1:4]` | `a[[1, 2, 3]]` |
| Returns | View (same memory) | Copy (new memory) |
| Order | Sequential only | Any order |

---

## 9. reshape()

`reshape()` changes the **shape (dimensions) of an array** without changing its data. The total number of elements must stay the same.

```python
a = np.arange(12)
print(a)         # [ 0  1  2  3  4  5  6  7  8  9 10 11]
print(a.shape)   # (12,)
```

### Reshape to 2D

```python
b = a.reshape(3, 4)   # 3 rows, 4 columns
print(b)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
```

### Reshape to 3D

```python
c = a.reshape(2, 3, 2)   # 2 blocks, 3 rows, 2 columns
print(c)
# [[[ 0  1]
#   [ 2  3]
#   [ 4  5]]
#  [[ 6  7]
#   [ 8  9]
#   [10 11]]]
```

### Using -1 (Auto-calculate one dimension)

When you don't want to calculate a dimension manually, use `-1` — NumPy figures it out.

```python
a = np.arange(12)

print(a.reshape(3, -1))    # shape (3, 4) → NumPy calculates 12/3 = 4
print(a.reshape(-1, 6))    # shape (2, 6) → NumPy calculates 12/6 = 2
print(a.reshape(2, 2, -1)) # shape (2, 2, 3)
```

### ⚠️ Rule: Total Elements Must Match

```python
a = np.arange(12)
a.reshape(5, 3)   # ❌ Error! 5×3=15 ≠ 12
a.reshape(4, 3)   # ✅ OK!    4×3=12 == 12
```

---

## 10. flatten() vs ravel()

Both convert a multi-dimensional array into a **1D (flat) array**. The difference is in memory.

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])
```

### flatten() — Always returns a COPY

```python
flat = a.flatten()
print(flat)      # [1 2 3 4 5 6]

flat[0] = 999    # Modify the copy
print(a)         # [[1 2 3] [4 5 6]]  ← original unchanged
```

### ravel() — Returns a VIEW (when possible)

```python
rav = a.ravel()
print(rav)       # [1 2 3 4 5 6]

rav[0] = 999     # Modify the view
print(a)         # [[999  2  3] [ 4  5  6]]  ← original changed!
```

### Comparison Table

| Feature | `flatten()` | `ravel()` |
|---------|-------------|-----------|
| Returns | Always a copy | View if possible, copy otherwise |
| Memory | More (new array) | Less (shares memory) |
| Safe to modify | ✅ Yes | ⚠️ Modifies original |
| Speed | Slightly slower | Slightly faster |

### When to use which?
- Use `flatten()` when you want a **safe independent copy**.
- Use `ravel()` when you want **speed and memory efficiency** and don't need to modify it.

---

## 11. transpose()

`transpose()` (or `.T`) **swaps rows and columns** — it flips the array along its diagonal.

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

print("Original shape:", a.shape)   # (2, 3)

b = a.T
print("Transposed shape:", b.shape) # (3, 2)
print(b)
# [[1 4]
#  [2 5]
#  [3 6]]
```

### Visual Explanation

```
Original (2×3):        Transposed (3×2):
[1  2  3]              [1  4]
[4  5  6]              [2  5]
                       [3  6]
```

### Using np.transpose() with custom axis order

For 3D arrays you can specify which axes to swap:

```python
a = np.arange(24).reshape(2, 3, 4)  # shape (2, 3, 4)

b = np.transpose(a, (1, 0, 2))      # swap axis 0 and 1
print(b.shape)                       # (3, 2, 4)
```

### Common Use Cases
- **Machine learning**: images are often stored as `(H, W, C)` — transpose lets you convert to `(C, H, W)`
- **Matrix math**: transpose is needed for matrix multiplication rules
- **Data formatting**: convert row data to column data or vice versa

---

## 12. Views vs Copies

This is one of the most **important concepts** to understand — it affects whether modifying a slice changes the original array or not.

### View — Shares memory with original

```python
a = np.array([1, 2, 3, 4, 5])

view = a[1:4]     # slicing returns a VIEW
view[0] = 999

print(view)   # [999   3   4]
print(a)      # [  1 999   3   4   5]  ← original changed!
```

### Copy — Independent from original

```python
a = np.array([1, 2, 3, 4, 5])

copy = a[1:4].copy()   # explicitly make a copy
copy[0] = 999

print(copy)   # [999   3   4]
print(a)      # [1 2 3 4 5]   ← original unchanged!
```

### How to check if it's a view

```python
b = a[:]
print(b.base is a)   # True  → it's a view

c = a.copy()
print(c.base is a)   # False → it's a copy
```

### Summary

| Operation | View or Copy? |
|-----------|---------------|
| `a[1:3]` (slicing) | View |
| `a[[0, 1]]` (fancy indexing) | Copy |
| `a[a > 5]` (boolean indexing) | Copy |
| `a.reshape(...)` | View (usually) |
| `a.ravel()` | View (usually) |
| `a.flatten()` | Copy (always) |
| `a.copy()` | Copy (always) |

---

## 13. Practice Exercises

Try solving these on your own before checking the answers!

### Exercise 1 — Indexing
```python
a = np.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])
# Q1: Get the value 25
# Q2: Get the entire last row
# Q3: Get the value at the last row, last column using negative indexing
```

<details>
<summary>Show Answers</summary>

```python
print(a[1, 1])     # 25
print(a[-1, :])    # [35 40 45]
print(a[-1, -1])   # 45
```
</details>

---

### Exercise 2 — Slicing
```python
a = np.arange(1, 26).reshape(5, 5)
# Q1: Extract the top-left 3×3 sub-matrix
# Q2: Get every other row, all columns
# Q3: Reverse the entire array
```

<details>
<summary>Show Answers</summary>

```python
print(a[:3, :3])     # top-left 3×3
print(a[::2, :])     # every other row
print(a[::-1, ::-1]) # reversed
```
</details>

---

### Exercise 3 — Boolean Indexing
```python
a = np.array([3, 15, 7, 22, 1, 18, 9, 30])
# Q1: Get all values greater than 10
# Q2: Replace all values less than 5 with 0
# Q3: Get values between 5 and 20 (inclusive)
```

<details>
<summary>Show Answers</summary>

```python
print(a[a > 10])                      # [15 22 18 30]
a[a < 5] = 0                          # [ 0 15  7 22  0 18  9 30]
print(a[(a >= 5) & (a <= 20)])        # [15  7 18  9]
```
</details>

---

### Exercise 4 — Reshape
```python
a = np.arange(24)
# Q1: Reshape to (6, 4)
# Q2: Reshape to (2, 3, 4)
# Q3: Reshape to (4, -1) — what shape does this give?
```

<details>
<summary>Show Answers</summary>

```python
print(a.reshape(6, 4))       # (6, 4)
print(a.reshape(2, 3, 4))    # (2, 3, 4)
print(a.reshape(4, -1).shape) # (4, 6)
```
</details>

---

## 14. Cheat Sheet

```python
import numpy as np

a = np.arange(24).reshape(4, 6)

# ── INDEXING ──────────────────────────────────────────────
a[0]           # first row (1D)
a[0, 2]        # row 0, col 2
a[-1, -1]      # last row, last col

# ── SLICING ───────────────────────────────────────────────
a[1:3]         # rows 1 and 2
a[:, 2:5]      # all rows, cols 2–4
a[::2, ::2]    # every other row and col
a[::-1]        # rows reversed

# ── BOOLEAN INDEXING ──────────────────────────────────────
a[a > 10]                    # elements > 10
a[(a > 5) & (a < 15)]        # between 5 and 15
a[a % 2 == 0]                # even numbers

# ── FANCY INDEXING ────────────────────────────────────────
a[[0, 2, 3]]                 # select rows 0, 2, 3
a[[0, 1], [2, 4]]            # select (0,2) and (1,4)

# ── RESHAPING ─────────────────────────────────────────────
a.reshape(6, 4)              # new shape (must match total)
a.reshape(2, -1)             # auto-calculate last dimension
a.flatten()                  # 1D copy
a.ravel()                    # 1D view
a.T                          # transpose
np.transpose(a, (1, 0))      # explicit axis swap

# ── VIEWS vs COPIES ───────────────────────────────────────
view = a[1:3]                # view — shares memory
copy = a[1:3].copy()         # copy — independent
```

---

## 🔗 What's Next?

After mastering Day 2, you're ready for:

➡️ **Day 3 — Broadcasting & Vectorized Operations**
Learn how NumPy automatically handles math between arrays of different shapes, and how to write blazing-fast loop-free code.

---

## 📚 Resources

- [NumPy Indexing Docs](https://numpy.org/doc/stable/user/basics.indexing.html)
- [NumPy Array Manipulation](https://numpy.org/doc/stable/reference/routines.array-manipulation.html)
- [Practice on Google Colab](https://colab.research.google.com/)

---

*Part of the [7-Day NumPy Learning Plan](./README.md) · Day 2 of 7*
