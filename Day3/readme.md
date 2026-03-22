# 📘 Day 3 — NumPy Broadcasting & Vectorized Operations (A to Z Guide)

A complete, beginner-friendly deep dive into writing fast, loop-free numerical code with NumPy.

---

## 📋 Table of Contents

1. [Why Avoid Python Loops?](#1-why-avoid-python-loops)
2. [Vectorized Operations — The Basics](#2-vectorized-operations--the-basics)
3. [Arithmetic Operations on Arrays](#3-arithmetic-operations-on-arrays)
4. [Comparison Operations](#4-comparison-operations)
5. [What is Broadcasting?](#5-what-is-broadcasting)
6. [Broadcasting Rules (Step by Step)](#6-broadcasting-rules-step-by-step)
7. [Broadcasting Examples](#7-broadcasting-examples)
8. [Universal Functions (ufuncs)](#8-universal-functions-ufuncs)
9. [Math ufuncs](#9-math-ufuncs)
10. [Trigonometric ufuncs](#10-trigonometric-ufuncs)
11. [Exponential & Logarithm ufuncs](#11-exponential--logarithm-ufuncs)
12. [Rounding ufuncs](#12-rounding-ufuncs)
13. [np.where() — Conditional Vectorization](#13-npwhere--conditional-vectorization)
14. [np.vectorize() — Vectorizing Custom Functions](#14-npvectorize--vectorizing-custom-functions)
15. [Performance: Loops vs Vectorization](#15-performance-loops-vs-vectorization)
16. [Practice Exercises](#16-practice-exercises)
17. [Cheat Sheet](#17-cheat-sheet)

---

## 1. Why Avoid Python Loops?

In regular Python, when you want to add 1 to every element in a list, you use a loop:

```python
data = [1, 2, 3, 4, 5]

# Python loop — slow!
result = []
for x in data:
    result.append(x + 1)

print(result)  # [2, 3, 4, 5, 6]
```

This works, but it's **slow** — especially for large datasets with millions of values.

NumPy solves this by doing the operation on the **entire array at once**, using optimized C code under the hood:

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])

# NumPy vectorized — fast!
result = a + 1
print(result)  # [2 3 4 5 6]
```

### Speed Comparison

For an array of 1 million elements:

| Approach | Time |
|----------|------|
| Python loop | ~500 ms |
| NumPy vectorized | ~2 ms |

**NumPy is ~250× faster.** This is because NumPy operations are implemented in C and operate on contiguous memory blocks — no Python overhead per element.

---

## 2. Vectorized Operations — The Basics

A **vectorized operation** applies a function or operator to **every element** of an array simultaneously, without any explicit loop.

```python
a = np.array([10, 20, 30, 40, 50])

# These all work element-by-element automatically
print(a + 5)    # [15 25 35 45 55]
print(a * 2)    # [20 40 60 80 100]
print(a ** 2)   # [100 400 900 1600 2500]
print(a / 10)   # [1. 2. 3. 4. 5.]
```

### Element-wise Operations on Two Arrays

When two arrays have the **same shape**, operations happen element by element:

```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

print(a + b)   # [11 22 33]  → 1+10, 2+20, 3+30
print(a * b)   # [10 40 90]  → 1×10, 2×20, 3×30
print(b - a)   # [ 9 18 27]  → 10-1, 20-2, 30-3
print(a / b)   # [0.1 0.1 0.1]
```

### Visual Explanation

```
a = [  1,   2,   3 ]
      +    +    +
b = [ 10,  20,  30 ]
      =    =    =
    [ 11,  22,  33 ]
```

---

## 3. Arithmetic Operations on Arrays

NumPy supports all standard arithmetic operators. You can use the **operator symbol** or the **NumPy function** — both give the same result.

| Operation | Operator | NumPy Function | Example |
|-----------|----------|----------------|---------|
| Addition | `a + b` | `np.add(a, b)` | `[1,2] + [3,4]` → `[4,6]` |
| Subtraction | `a - b` | `np.subtract(a, b)` | `[5,6] - [1,2]` → `[4,4]` |
| Multiplication | `a * b` | `np.multiply(a, b)` | `[2,3] * [4,5]` → `[8,15]` |
| Division | `a / b` | `np.divide(a, b)` | `[6,8] / [2,4]` → `[3.,2.]` |
| Floor Division | `a // b` | `np.floor_divide(a, b)` | `[7,9] // [2,4]` → `[3,2]` |
| Modulus | `a % b` | `np.mod(a, b)` | `[7,9] % [3,4]` → `[1,1]` |
| Power | `a ** b` | `np.power(a, b)` | `[2,3] ** [3,2]` → `[8,9]` |

```python
a = np.array([10, 20, 30])
b = np.array([3, 4, 5])

print(a + b)    # [13 24 35]
print(a - b)    # [ 7 16 25]
print(a * b)    # [ 30  80 150]
print(a / b)    # [3.33 5.   6.  ]
print(a // b)   # [3 5 6]
print(a % b)    # [1 0 0]
print(a ** 2)   # [100 400 900]
```

### Operations with Scalars

A **scalar** is a single number. NumPy automatically applies it to every element:

```python
a = np.array([1, 2, 3, 4, 5])

print(a + 100)   # [101 102 103 104 105]
print(a * 3)     # [ 3  6  9 12 15]
print(a ** 2)    # [ 1  4  9 16 25]
print(10 / a)    # [10.   5.   3.33 2.5  2.  ]
```

This is actually the simplest case of **broadcasting** — the scalar gets "stretched" to match the array shape.

---

## 4. Comparison Operations

Comparison operators also work element-wise and return a **boolean array**:

```python
a = np.array([1, 5, 3, 8, 2, 7])

print(a > 4)       # [False  True False  True False  True]
print(a == 3)      # [False False  True False False False]
print(a != 5)      # [ True False  True  True  True  True]
print(a >= 5)      # [False  True False  True False  True]
```

### Comparing Two Arrays

```python
a = np.array([1, 5, 3])
b = np.array([2, 5, 1])

print(a > b)    # [False False  True]
print(a == b)   # [False  True False]
```

### Logical Operators

Combine multiple conditions using `&` (AND), `|` (OR), `~` (NOT):

```python
a = np.array([1, 5, 3, 8, 2, 7])

print(a[(a > 3) & (a < 7)])   # [5]        → between 3 and 7
print(a[(a < 2) | (a > 6)])   # [1 8 7]    → less than 2 or greater than 6
print(a[~(a > 5)])             # [1 3 2]    → NOT greater than 5
```

### Useful Functions

```python
a = np.array([True, False, True, True])

np.any(a)    # True  → at least one True
np.all(a)    # False → not all True
np.sum(a)    # 3     → count of True values (True = 1, False = 0)
```

---

## 5. What is Broadcasting?

Broadcasting is NumPy's powerful mechanism that allows **arithmetic operations between arrays of different shapes**.

Without broadcasting, you could only do math between arrays of the **exact same shape**. Broadcasting removes that restriction by "virtually stretching" the smaller array to match the larger one — **without actually copying any data**.

### Simple Example

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])        # shape (2, 3)

b = np.array([10, 20, 30])      # shape (3,)

result = a + b
# [[11 22 33]
#  [14 25 36]]
```

What happened? `b` was **broadcast** (stretched) across each row of `a`:

```
a = [[ 1,  2,  3],      b = [10, 20, 30]
     [ 4,  5,  6]]

Broadcasting stretches b to:
     [[10, 20, 30],
      [10, 20, 30]]

Result:
     [[11, 22, 33],
      [14, 25, 36]]
```

No data was actually copied — NumPy does this virtually for efficiency.

---

## 6. Broadcasting Rules (Step by Step)

NumPy follows 3 strict rules to decide if two arrays can be broadcast together.

### The Rules

**Rule 1:** If arrays have different numbers of dimensions, the shape of the smaller one is **padded with 1s on the LEFT**.

**Rule 2:** Arrays with size 1 along a dimension are **stretched** to match the other array's size in that dimension.

**Rule 3:** If two dimensions are neither equal nor one of them is 1, broadcasting **fails** with an error.

---

### Rule 1 in Action

```
Array A shape: (4, 3)
Array B shape:    (3,)   ← only 1D

After padding with 1 on the left:
Array B shape: (1, 3)
```

### Rule 2 in Action

```
Array A shape: (4, 3)
Array B shape: (1, 3)   ← size 1 in dim 0, stretched to 4

Final shape:   (4, 3)   ✅ Compatible!
```

### Rule 3 — When Broadcasting Fails

```
Array A shape: (3, 4)
Array B shape: (3,)
After padding: (1, 3)

Check: dim 0 → 3 vs 1 → OK (stretch 1 to 3)
       dim 1 → 4 vs 3 → ❌ Neither is 1 → ERROR!
```

---

### Quick Compatibility Check

```
Shape A:  (8, 1, 6, 1)
Shape B:     (7, 1, 5)

After padding B:
Shape B:  (1, 7, 1, 5)

Check each dim:
  8 vs 1 → 8  ✅
  1 vs 7 → 7  ✅
  6 vs 1 → 6  ✅
  1 vs 5 → 5  ✅

Result:   (8, 7, 6, 5) ✅ Compatible!
```

---

## 7. Broadcasting Examples

### Example 1 — Scalar broadcast (simplest)

```python
a = np.array([1, 2, 3])
print(a + 10)   # [11 12 13]
# 10 is broadcast to [10, 10, 10] then added
```

### Example 2 — 1D to 2D (add to each row)

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])        # shape (2, 3)
b = np.array([10, 20, 30])      # shape (3,)

print(a + b)
# [[11 22 33]
#  [14 25 36]]
```

### Example 3 — Column vector to 2D (add to each column)

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])        # shape (2, 3)
b = np.array([[10],
              [20]])             # shape (2, 1)

print(a + b)
# [[11 12 13]
#  [24 25 26]]
```

### Example 4 — Both arrays broadcast (outer addition)

```python
row = np.array([0, 10, 20, 30])    # shape (4,)  → becomes (1, 4)
col = np.array([[0],               # shape (3, 1)
                [1],
                [2]])

print(row + col)
# [[ 0 10 20 30]
#  [ 1 11 21 31]
#  [ 2 12 22 32]]
```

### Example 5 — Normalizing rows of a matrix

```python
data = np.array([[10, 20, 30],
                 [40, 50, 60],
                 [70, 80, 90]])

row_means = data.mean(axis=1, keepdims=True)  # shape (3, 1)
print(row_means)
# [[20.]
#  [50.]
#  [80.]]

normalized = data - row_means   # broadcast subtracts each row's mean
print(normalized)
# [[-10.   0.  10.]
#  [-10.   0.  10.]
#  [-10.   0.  10.]]
```

### ⚠️ Common Broadcasting Mistake

```python
a = np.array([1, 2, 3])     # shape (3,)
b = np.array([1, 2])        # shape (2,)

a + b  # ❌ Error! 3 vs 2, neither is 1

# Fix: reshape b to (2, 1) or make shapes compatible
```

---

## 8. Universal Functions (ufuncs)

A **ufunc** (universal function) is a NumPy function that operates **element-wise** on arrays. They are implemented in C, making them extremely fast.

ufuncs are the building blocks of vectorized computing in NumPy.

```python
a = np.array([1, 4, 9, 16, 25])

# ufunc applies to every element at once
print(np.sqrt(a))    # [1. 2. 3. 4. 5.]
```

### Why ufuncs vs Python math functions?

```python
import math

a = np.array([1, 4, 9])

# ❌ Python math — works on scalars only
math.sqrt(a)         # TypeError!

# ✅ NumPy ufunc — works on arrays
np.sqrt(a)           # [1. 2. 3.]
```

### ufunc Categories

| Category | Functions |
|----------|-----------|
| Math | `np.abs`, `np.sqrt`, `np.square`, `np.sign` |
| Trig | `np.sin`, `np.cos`, `np.tan`, `np.arcsin` |
| Exp/Log | `np.exp`, `np.exp2`, `np.log`, `np.log2`, `np.log10` |
| Rounding | `np.floor`, `np.ceil`, `np.round`, `np.trunc` |
| Comparison | `np.maximum`, `np.minimum`, `np.clip` |

---

## 9. Math ufuncs

```python
a = np.array([-3, -1, 0, 1, 4, 9])

# Absolute value
print(np.abs(a))        # [3 1 0 1 4 9]

# Square root (only valid for non-negative numbers)
print(np.sqrt(np.abs(a)))  # [1.73 1.   0.   1.   2.   3.  ]

# Square (element-wise x²)
print(np.square(a))     # [ 9  1  0  1 16 81]

# Sign: returns -1, 0, or 1
print(np.sign(a))       # [-1 -1  0  1  1  1]

# Reciprocal: 1/x
print(np.reciprocal(np.array([1., 2., 4.])))   # [1.   0.5  0.25]
```

### np.maximum and np.minimum

```python
a = np.array([1, 5, 3])
b = np.array([4, 2, 6])

print(np.maximum(a, b))   # [4 5 6]  → element-wise max
print(np.minimum(a, b))   # [1 2 3]  → element-wise min
```

### np.clip — Clamp values to a range

```python
a = np.array([2, 8, -1, 15, 5])

print(np.clip(a, 0, 10))   # [ 2  8  0 10  5]
# Values below 0 → 0, values above 10 → 10
```

---

## 10. Trigonometric ufuncs

NumPy trig functions use **radians** by default. Use `np.deg2rad()` to convert degrees.

```python
angles_deg = np.array([0, 30, 45, 60, 90])
angles_rad = np.deg2rad(angles_deg)

print(np.sin(angles_rad))
# [0.    0.5   0.707 0.866 1.   ]

print(np.cos(angles_rad))
# [1.    0.866 0.707 0.5   0.   ]

print(np.tan(angles_rad))
# [0.    0.577 1.    1.732 ∞   ]
```

### Inverse trig

```python
a = np.array([0, 0.5, 1])

print(np.arcsin(a))    # [0.    0.524 1.571]  → in radians
print(np.rad2deg(np.arcsin(a)))   # [ 0.  30.  90.]  → in degrees
```

### Useful constants

```python
print(np.pi)   # 3.141592653589793
print(np.e)    # 2.718281828459045
```

---

## 11. Exponential & Logarithm ufuncs

### Exponential

```python
a = np.array([0, 1, 2, 3])

print(np.exp(a))     # [ 1.     2.718  7.389 20.086]  → e^x
print(np.exp2(a))    # [ 1.     2.     4.     8.   ]  → 2^x
```

### Logarithm

```python
a = np.array([1, np.e, 10, 100])

print(np.log(a))     # [0.    1.    2.303 4.605]  → natural log (base e)
print(np.log2(a))    # [0.    1.443 3.322 6.644]  → log base 2
print(np.log10(a))   # [0.    0.434 1.    2.   ]  → log base 10
```

### Real-world use: compound interest

```python
principal = np.array([1000, 5000, 10000])
rate = 0.05
years = 10

# A = P * e^(r*t)  (continuous compounding)
amount = principal * np.exp(rate * years)
print(np.round(amount, 2))
# [ 1648.72  8243.61 16487.21]
```

---

## 12. Rounding ufuncs

```python
a = np.array([1.2, 1.5, 1.7, 2.5, -1.3, -1.7])

print(np.floor(a))    # [ 1.  1.  1.  2. -2. -2.]  → round DOWN
print(np.ceil(a))     # [ 2.  2.  2.  3. -1. -1.]  → round UP
print(np.trunc(a))    # [ 1.  1.  1.  2. -1. -1.]  → truncate (toward zero)
print(np.round(a))    # [ 1.  2.  2.  2. -1. -2.]  → round to nearest even
print(np.round(a, 1)) # rounds to 1 decimal place
```

### Difference between floor, ceil, trunc

```
Value:   1.7     -1.7
floor:   1.0     -2.0   (always goes DOWN on number line)
ceil:    2.0     -1.0   (always goes UP on number line)
trunc:   1.0     -1.0   (always goes toward ZERO)
```

---

## 13. np.where() — Conditional Vectorization

`np.where()` is the vectorized version of an `if-else` statement. It selects elements from two arrays based on a condition.

### Syntax
```python
np.where(condition, value_if_true, value_if_false)
```

### Basic Example

```python
a = np.array([10, -5, 30, -2, 15])

result = np.where(a > 0, a, 0)   # keep positives, replace negatives with 0
print(result)   # [10  0 30  0 15]
```

### Replace with constants

```python
a = np.array([1, 2, 3, 4, 5, 6])

result = np.where(a % 2 == 0, "even", "odd")
print(result)   # ['odd' 'even' 'odd' 'even' 'odd' 'even']
```

### Nested np.where (like if-elif-else)

```python
scores = np.array([85, 45, 72, 91, 55])

grades = np.where(scores >= 90, 'A',
         np.where(scores >= 75, 'B',
         np.where(scores >= 60, 'C', 'F')))

print(grades)   # ['B' 'F' 'C' 'A' 'F']
```

### np.where() with just condition (returns indices)

```python
a = np.array([10, -5, 30, -2, 15])

indices = np.where(a > 0)
print(indices)          # (array([0, 2, 4]),)
print(a[indices])       # [10 30 15]
```

---

## 14. np.vectorize() — Vectorizing Custom Functions

Sometimes you have a regular Python function that works on a single value. `np.vectorize()` wraps it so it works on entire arrays.

```python
def classify(x):
    if x < 0:
        return "negative"
    elif x == 0:
        return "zero"
    else:
        return "positive"

# Without vectorize — only works on a single number
classify(5)      # "positive"
# classify(np.array([1, -2, 0]))  ← This would fail!

# With vectorize — works on entire array
vclassify = np.vectorize(classify)
a = np.array([3, -1, 0, 7, -5])
print(vclassify(a))   # ['positive' 'negative' 'zero' 'positive' 'negative']
```

### ⚠️ Important Note

`np.vectorize()` is **not** actually faster than a loop — it's just syntactic convenience. For real performance, always prefer built-in ufuncs or `np.where()`.

---

## 15. Performance: Loops vs Vectorization

Let's prove that vectorization is dramatically faster.

```python
import numpy as np
import time

n = 1_000_000
a = np.random.rand(n)
b = np.random.rand(n)

# Python loop
start = time.time()
result_loop = [a[i] + b[i] for i in range(n)]
loop_time = time.time() - start
print(f"Loop:   {loop_time*1000:.1f} ms")

# NumPy vectorized
start = time.time()
result_numpy = a + b
numpy_time = time.time() - start
print(f"NumPy:  {numpy_time*1000:.1f} ms")

print(f"Speedup: {loop_time / numpy_time:.0f}×")
```

**Typical output:**
```
Loop:   450.3 ms
NumPy:    1.8 ms
Speedup: 250×
```

### Rules for Maximizing Performance

| ✅ Do This | ❌ Avoid This |
|------------|--------------|
| Use `a + b` | `for x in a: x + b` |
| Use `np.sqrt(a)` | `[math.sqrt(x) for x in a]` |
| Use `np.where(cond, x, y)` | `[x if c else y for c in cond]` |
| Use `a.sum()` | `sum(a)` (Python built-in) |
| Use boolean masks | Manual filtering loops |

---

## 16. Practice Exercises

### Exercise 1 — Vectorized Arithmetic
```python
a = np.array([2, 4, 6, 8, 10])
b = np.array([1, 3, 5, 7, 9])

# Q1: Compute a² + b²
# Q2: Compute (a - b) / (a + b)
# Q3: Find which elements of a are divisible by 4
```

<details>
<summary>Show Answers</summary>

```python
print(a**2 + b**2)           # [  5  25  61 113 181]
print((a - b) / (a + b))     # [0.333 0.143 0.091 0.067 0.053]
print(a[a % 4 == 0])         # [ 4  8]
```
</details>

---

### Exercise 2 — Broadcasting
```python
matrix = np.arange(1, 10).reshape(3, 3)
# Q1: Add [1, 2, 3] to each row
# Q2: Multiply each column by [[2], [3], [4]]
# Q3: Subtract the mean of each row from that row (row normalization)
```

<details>
<summary>Show Answers</summary>

```python
print(matrix + np.array([1, 2, 3]))

print(matrix * np.array([[2], [3], [4]]))

row_means = matrix.mean(axis=1, keepdims=True)
print(matrix - row_means)
```
</details>

---

### Exercise 3 — ufuncs
```python
a = np.array([0, 30, 60, 90, 120, 180])
# Q1: Compute sine of each angle (convert degrees to radians first)
# Q2: Find e^x for x = [1, 2, 3, 4]
# Q3: Clip all values to range [30, 120]
```

<details>
<summary>Show Answers</summary>

```python
print(np.sin(np.deg2rad(a)))

x = np.array([1, 2, 3, 4])
print(np.exp(x))

print(np.clip(a, 30, 120))
```
</details>

---

### Exercise 4 — np.where()
```python
temperatures = np.array([15, 38, 22, 5, 42, 30, 18])
# Q1: Label each as 'hot' (>35), 'warm' (20-35), or 'cold' (<20)
# Q2: Replace all temperatures above 40 with 40 (cap the values)
# Q3: Find indices of 'cold' temperatures
```

<details>
<summary>Show Answers</summary>

```python
labels = np.where(temperatures > 35, 'hot',
         np.where(temperatures >= 20, 'warm', 'cold'))
print(labels)

capped = np.where(temperatures > 40, 40, temperatures)
print(capped)

cold_indices = np.where(temperatures < 20)
print(cold_indices)
```
</details>

---

## 17. Cheat Sheet

```python
import numpy as np

a = np.array([1, 4, 9, 16])
b = np.array([2, 2, 3, 4])

# ── ARITHMETIC ────────────────────────────────────────────
a + b              # element-wise addition
a - b              # subtraction
a * b              # multiplication
a / b              # division
a ** 2             # power
a % 3              # modulus
a // 3             # floor division

# ── COMPARISON ────────────────────────────────────────────
a > 5              # boolean array
a == b             # element-wise equality
(a > 2) & (a < 10) # AND condition
(a < 2) | (a > 10) # OR condition
~(a > 5)           # NOT condition

# ── BROADCASTING ──────────────────────────────────────────
a + 10             # scalar broadcast
m + np.array([1, 2, 3])        # row-wise broadcast
m + np.array([[1], [2], [3]])  # column-wise broadcast

# ── MATH ufuncs ───────────────────────────────────────────
np.abs(a)          # absolute value
np.sqrt(a)         # square root
np.square(a)       # element-wise square
np.sign(a)         # sign: -1, 0, or 1
np.clip(a, 0, 10)  # clamp to [0, 10]
np.maximum(a, b)   # element-wise max
np.minimum(a, b)   # element-wise min

# ── TRIG ufuncs ───────────────────────────────────────────
np.sin(a)          # sine (radians)
np.cos(a)          # cosine
np.tan(a)          # tangent
np.deg2rad(a)      # degrees → radians
np.rad2deg(a)      # radians → degrees
np.arcsin(a)       # inverse sine

# ── EXP / LOG ─────────────────────────────────────────────
np.exp(a)          # e^x
np.exp2(a)         # 2^x
np.log(a)          # natural log
np.log2(a)         # log base 2
np.log10(a)        # log base 10

# ── ROUNDING ──────────────────────────────────────────────
np.floor(a)        # round down
np.ceil(a)         # round up
np.trunc(a)        # truncate toward zero
np.round(a, 2)     # round to 2 decimal places

# ── CONDITIONAL ───────────────────────────────────────────
np.where(a > 5, a, 0)           # if a>5 keep, else 0
np.where(a > 5, 'big', 'small') # string labels
np.where(a > 5)                 # returns indices where True

# ── PERFORMANCE ───────────────────────────────────────────
# ✅ Always prefer vectorized operations over Python loops
# ✅ Use ufuncs (np.sqrt, np.exp) over math module functions
# ✅ Use np.where() instead of list comprehensions with conditions
```

---

## 🔗 What's Next?

After mastering Day 3, you're ready for:

➡️ **Day 4 — Aggregation & Statistics**
Learn to summarize data: sums, means, standard deviations, sorting, and working along specific axes.

---

## 📚 Resources

- [NumPy Broadcasting Docs](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [NumPy ufuncs Reference](https://numpy.org/doc/stable/reference/ufuncs.html)
- [NumPy for Absolute Beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [Practice on Google Colab](https://colab.research.google.com/)

---

*Part of the [7-Day NumPy Learning Plan](./README.md) · Day 3 of 7*
