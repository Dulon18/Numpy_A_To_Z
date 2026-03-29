# 📘 Day 5 — NumPy Linear Algebra & Random Module (A to Z Guide)

A complete, beginner-friendly deep dive into matrix operations, solving equations, eigenvalues, and simulating data with NumPy.

---

## 📋 Table of Contents

1. [What is Linear Algebra?](#1-what-is-linear-algebra)
2. [Creating Matrices](#2-creating-matrices)
3. [Matrix Arithmetic](#3-matrix-arithmetic)
4. [dot() and matmul() — Matrix Multiplication](#4-dot-and-matmul--matrix-multiplication)
5. [np.linalg.inv() — Matrix Inverse](#5-nplinalginv--matrix-inverse)
6. [np.linalg.det() — Determinant](#6-nplinalgdet--determinant)
7. [np.linalg.solve() — Solving Linear Equations](#7-nplinalgsolve--solving-linear-equations)
8. [np.linalg.eig() — Eigenvalues & Eigenvectors](#8-nplinalgeig--eigenvalues--eigenvectors)
9. [np.linalg.norm() — Vector & Matrix Norms](#9-nplinalgnorm--vector--matrix-norms)
10. [np.linalg.rank() and np.linalg.matrix_rank()](#10-nplinalgrank-and-nplinalgmatrix_rank)
11. [np.linalg.svd() — Singular Value Decomposition](#11-nplinalgsvd--singular-value-decomposition)
12. [np.linalg.lstsq() — Least Squares Solution](#12-nplinalglstsq--least-squares-solution)
13. [What is the Random Module?](#13-what-is-the-random-module)
14. [np.random.seed() — Reproducibility](#14-nprandomseed--reproducibility)
15. [Generating Random Numbers](#15-generating-random-numbers)
16. [Random Integers — randint()](#16-random-integers--randint)
17. [Random Choice — choice()](#17-random-choice--choice)
18. [Random Shuffle — shuffle() and permutation()](#18-random-shuffle--shuffle-and-permutation)
19. [Probability Distributions](#19-probability-distributions)
20. [Real-World Example — Simulation & Linear System](#20-real-world-example--simulation--linear-system)
21. [Practice Exercises](#21-practice-exercises)
22. [Cheat Sheet](#22-cheat-sheet)

---

## 1. What is Linear Algebra?

**Linear algebra** is the branch of mathematics dealing with **vectors** and **matrices** — and operations on them. It is the backbone of:

- Machine learning (neural networks, PCA, SVD)
- Computer graphics (3D transformations)
- Physics simulations
- Data science (regression, dimensionality reduction)
- Engineering (systems of equations)

In NumPy, the `np.linalg` module provides all the tools needed for linear algebra.

```
A vector is a 1D array:   [1, 2, 3]
A matrix is a 2D array:   [[1, 2],
                            [3, 4]]
```

---

## 2. Creating Matrices

Before doing linear algebra, you need to know how to create special matrices:

```python
import numpy as np

# Identity matrix — diagonal of 1s, rest 0s
I = np.eye(3)
print(I)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# Zero matrix
Z = np.zeros((3, 3))

# Ones matrix
O = np.ones((2, 4))

# Diagonal matrix — values on the main diagonal
D = np.diag([1, 2, 3])
print(D)
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]

# Extract diagonal from a matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(np.diag(A))    # [1 5 9]  → main diagonal values

# Upper and lower triangular
print(np.triu(A))    # upper triangle (zeros below diagonal)
print(np.tril(A))    # lower triangle (zeros above diagonal)
```

### Matrix from function

```python
# np.fromfunction builds matrix using index values
A = np.fromfunction(lambda i, j: i + j, (4, 4), dtype=int)
print(A)
# [[0 1 2 3]
#  [1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]]
```

---

## 3. Matrix Arithmetic

### Element-wise vs Matrix operations

This is a critical distinction. `*` does **element-wise** multiplication. Use `@` or `np.dot()` for true **matrix multiplication**.

```python
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

# Element-wise (NOT matrix multiplication)
print(A * B)
# [[ 5 12]
#  [21 32]]

# Matrix multiplication (dot product)
print(A @ B)
# [[19 22]
#  [43 50]]
```

### Element-wise Operations

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A + B)    # element-wise addition
print(A - B)    # element-wise subtraction
print(A * B)    # element-wise multiplication
print(A / B)    # element-wise division
print(A ** 2)   # element-wise square
```

### Transpose

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(A.T)
# [[1 4]
#  [2 5]
#  [3 6]]
```

### Scalar Operations

```python
A = np.array([[1, 2], [3, 4]])

print(A + 10)    # add 10 to every element
print(A * 3)     # multiply every element by 3
print(A / 2)     # divide every element by 2
```

---

## 4. dot() and matmul() — Matrix Multiplication

Matrix multiplication is the **core operation** in linear algebra. It is NOT the same as element-wise multiplication.

### How Matrix Multiplication Works

For `C = A @ B`:
- A has shape `(m, n)`
- B has shape `(n, p)`
- C has shape `(m, p)`
- The **inner dimensions must match**: `n == n`

```
A (2×3) × B (3×2) = C (2×2)   ✅
A (2×3) × B (2×3) = ERROR      ❌ (inner dims 3 ≠ 2)
```

Each element `C[i,j]` = dot product of row i of A and column j of B:

```
A = [[1, 2, 3],      B = [[7, 8],
     [4, 5, 6]]           [9, 10],
                           [11, 12]]

C[0,0] = 1×7 + 2×9 + 3×11 = 7+18+33 = 58
C[0,1] = 1×8 + 2×10 + 3×12 = 8+20+36 = 64
C[1,0] = 4×7 + 5×9 + 6×11 = 28+45+66 = 139
C[1,1] = 4×8 + 5×10 + 6×12 = 32+50+72 = 154
```

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

B = np.array([[7,  8],
              [9,  10],
              [11, 12]])

# Three equivalent ways to do matrix multiplication:
print(np.dot(A, B))       # method 1
print(np.matmul(A, B))    # method 2
print(A @ B)              # method 3 (preferred, cleanest)

# Output:
# [[ 58  64]
#  [139 154]]
```

### dot() for 1D vectors — Dot Product

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Dot product = sum of element-wise products
print(np.dot(a, b))   # 1×4 + 2×5 + 3×6 = 4+10+18 = 32
```

### Difference: dot() vs matmul()

| Situation | `np.dot()` | `np.matmul()` / `@` |
|-----------|------------|----------------------|
| 2D arrays | Matrix multiply | Matrix multiply (same) |
| 1D arrays | Dot product (scalar) | Dot product (same) |
| Scalar input | Scales array | ❌ Not allowed |
| 3D+ arrays | Different behavior | Batch matrix multiply |

Use `@` (matmul) as the **default** for matrix multiplication — it's cleaner and more predictable.

### Batch Matrix Multiplication (3D)

```python
# 5 matrices of shape (3, 3) multiplied with 5 matrices of (3, 2)
A = np.random.rand(5, 3, 3)
B = np.random.rand(5, 3, 2)

C = A @ B
print(C.shape)   # (5, 3, 2) → 5 result matrices of shape (3, 2)
```

---

## 5. np.linalg.inv() — Matrix Inverse

The **inverse** of a matrix `A` is a matrix `A⁻¹` such that `A @ A⁻¹ = I` (identity matrix).

Only **square matrices** with **non-zero determinant** have an inverse.

```python
A = np.array([[1., 2.],
              [3., 4.]])

A_inv = np.linalg.inv(A)
print(A_inv)
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# Verify: A @ A_inv should equal identity matrix
print(np.round(A @ A_inv, 10))
# [[1. 0.]
#  [0. 1.]]
```

### When Does Inverse NOT Exist?

A matrix is **singular** (non-invertible) if its determinant is 0:

```python
A = np.array([[1., 2.],
              [2., 4.]])   # row 2 = 2× row 1 (linearly dependent)

try:
    np.linalg.inv(A)
except np.linalg.LinAlgError as e:
    print(f"Error: {e}")   # Singular matrix
```

### Real-World Use — Solving Ax = b

If `A @ x = b`, then `x = A⁻¹ @ b`:

```python
A = np.array([[2., 1.],
              [5., 3.]])
b = np.array([4., 7.])

x = np.linalg.inv(A) @ b
print(x)   # [5. -6.]

# Better: use np.linalg.solve() for this (more stable numerically)
```

---

## 6. np.linalg.det() — Determinant

The **determinant** is a single number that tells you:
- Whether a matrix has an inverse (`det ≠ 0` → invertible)
- How much a matrix scales area/volume (geometric interpretation)
- Whether rows/columns are linearly independent

```python
A = np.array([[1., 2.],
              [3., 4.]])

print(np.linalg.det(A))   # -2.0
# det([[a,b],[c,d]]) = a*d - b*c = 1*4 - 2*3 = -2
```

### Interpreting the Determinant

| det value | Meaning |
|-----------|---------|
| `det = 0` | Singular matrix, no inverse, rows are linearly dependent |
| `det = 1` | Rotation matrix (preserves area) |
| `\|det\| > 1` | Matrix scales (expands) area |
| `\|det\| < 1` | Matrix shrinks area |
| `det < 0` | Matrix flips orientation |

```python
# Identity matrix — det = 1
print(np.linalg.det(np.eye(3)))   # 1.0

# Singular matrix — det = 0
A = np.array([[1., 2.], [2., 4.]])
print(np.linalg.det(A))           # 0.0  → no inverse!

# Scaling matrix
A = np.array([[2., 0.], [0., 3.]])
print(np.linalg.det(A))           # 6.0  → scales area by factor 6
```

### 3×3 Determinant

```python
A = np.array([[1., 2., 3.],
              [4., 5., 6.],
              [7., 8., 10.]])

print(np.linalg.det(A))   # -3.0
```

---

## 7. np.linalg.solve() — Solving Linear Equations

`np.linalg.solve(A, b)` solves the system of linear equations `A @ x = b` for unknown vector `x`.

This is **more numerically stable** than computing `inv(A) @ b`.

### Example: System of 2 equations

```
2x + y  = 8
x  + 3y = 11
```

In matrix form: `A @ x = b`
```
A = [[2, 1],     b = [8,
     [1, 3]]          11]
```

```python
A = np.array([[2., 1.],
              [1., 3.]])
b = np.array([8., 11.])

x = np.linalg.solve(A, b)
print(x)   # [3. 2.]
# Solution: x=3, y=2

# Verify: A @ x should equal b
print(np.allclose(A @ x, b))   # True
```

### Example: System of 3 equations

```
3x +  y -  z = 5
x  - 2y + 3z = 2
2x +  y + 2z = 9
```

```python
A = np.array([[ 3.,  1., -1.],
              [ 1., -2.,  3.],
              [ 2.,  1.,  2.]])
b = np.array([5., 2., 9.])

x = np.linalg.solve(A, b)
print(np.round(x, 4))   # [2. 1. 2.]
# x=2, y=1, z=2

print(np.allclose(A @ x, b))   # True ✅
```

### solve() vs inv() — Which to Use?

```python
# ❌ Less stable — uses matrix inverse
x = np.linalg.inv(A) @ b

# ✅ More stable — uses LU decomposition internally
x = np.linalg.solve(A, b)
```

Always prefer `solve()` over `inv() @ b` for solving linear systems.

### Multiple Right-Hand Sides

```python
A = np.array([[2., 1.],
              [1., 3.]])

# Solve for two different b vectors at once
B = np.array([[8., 10.],
              [11., 13.]])

X = np.linalg.solve(A, B)
print(X)   # each column is the solution for the corresponding column of B
```

---

## 8. np.linalg.eig() — Eigenvalues & Eigenvectors

For a square matrix `A`, an **eigenvector** `v` and **eigenvalue** `λ` satisfy:

```
A @ v = λ × v
```

This means: multiplying matrix A by vector v only **scales** v by λ — it doesn't change its direction.

Eigenvalues/vectors are used in:
- **PCA** (Principal Component Analysis) — finding directions of maximum variance
- **Google PageRank** — ranking web pages
- **Vibration analysis** — finding resonant frequencies
- **Quantum mechanics** — energy states

```python
A = np.array([[4., 1.],
              [2., 3.]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
# [5. 2.]

print("Eigenvectors (columns):")
print(eigenvectors)
# [[ 0.707 -0.447]
#  [ 0.707  0.894]]
# Column 0 is eigenvector for eigenvalue 5
# Column 1 is eigenvector for eigenvalue 2
```

### Verification

```python
# For each eigenvalue/eigenvector pair: A @ v = λ × v
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v   = eigenvectors[:, i]   # column i is the eigenvector

    Av     = A @ v
    lam_v  = lam * v

    print(f"λ={lam:.1f}: A@v = {np.round(Av, 4)}, λ×v = {np.round(lam_v, 4)}")
    print(f"  Match: {np.allclose(Av, lam_v)}")

# λ=5.0: A@v = [3.536 3.536], λ×v = [3.536 3.536]  Match: True
# λ=2.0: A@v = [-0.894 1.789], λ×v = [-0.894 1.789]  Match: True
```

### np.linalg.eigvals() — Only eigenvalues (faster)

```python
A = np.array([[4., 1.], [2., 3.]])
print(np.linalg.eigvals(A))   # [5. 2.]
```

### Symmetric matrices — np.linalg.eigh()

For **symmetric** matrices (A = Aᵀ), use `eigh()` — it's faster and guarantees real eigenvalues:

```python
A = np.array([[4., 2.],
              [2., 3.]])   # symmetric!

values, vectors = np.linalg.eigh(A)
print(values)   # [1.438 5.562]  → always real
```

---

## 9. np.linalg.norm() — Vector & Matrix Norms

A **norm** is a measure of the **size** or **length** of a vector or matrix.

### Vector Norms

```python
v = np.array([3., 4.])

# L2 norm (Euclidean distance, default) = sqrt(3²+4²) = 5
print(np.linalg.norm(v))        # 5.0

# L1 norm (Manhattan distance) = |3| + |4| = 7
print(np.linalg.norm(v, ord=1)) # 7.0

# L∞ norm (max absolute value)
print(np.linalg.norm(v, ord=np.inf))  # 4.0
```

### L2 Norm Formula

```
||v||₂ = sqrt(v[0]² + v[1]² + ... + v[n]²)
```

```python
# Manual calculation to understand
v = np.array([1., 2., 3., 4.])
print(np.sqrt(np.sum(v**2)))     # 5.477
print(np.linalg.norm(v))         # 5.477  ← same result
```

### Normalizing a Vector (Unit Vector)

A **unit vector** has length 1. To normalize:

```python
v = np.array([3., 4.])

unit_v = v / np.linalg.norm(v)
print(unit_v)                      # [0.6 0.8]
print(np.linalg.norm(unit_v))      # 1.0  ✅
```

### Matrix Norms

```python
A = np.array([[1., 2.],
              [3., 4.]])

print(np.linalg.norm(A))           # Frobenius norm (default) = sqrt(1+4+9+16) = 5.477
print(np.linalg.norm(A, 'fro'))    # same
print(np.linalg.norm(A, 1))        # max column sum
print(np.linalg.norm(A, np.inf))   # max row sum
```

### Distance Between Two Vectors

```python
a = np.array([1., 2., 3.])
b = np.array([4., 6., 3.])

distance = np.linalg.norm(a - b)
print(distance)   # 5.0  → Euclidean distance
```

---

## 10. np.linalg.rank() and np.linalg.matrix_rank()

The **rank** of a matrix is the number of **linearly independent rows (or columns)**.

```python
# Full rank matrix (rank = min(rows, cols))
A = np.array([[1., 2., 3.],
              [4., 5., 6.],
              [7., 8., 10.]])
print(np.linalg.matrix_rank(A))   # 3  → full rank

# Rank-deficient matrix (rows are not independent)
B = np.array([[1., 2., 3.],
              [2., 4., 6.],   # row 2 = 2 × row 1
              [3., 6., 9.]])  # row 3 = 3 × row 1
print(np.linalg.matrix_rank(B))   # 1  → only 1 independent row
```

### What rank tells you

| Rank | Meaning |
|------|---------|
| Full rank | All rows/columns are independent; unique solution exists |
| Rank deficient | Rows/columns are linearly dependent; no unique solution |
| Rank = 0 | Zero matrix |

---

## 11. np.linalg.svd() — Singular Value Decomposition

**SVD** decomposes any matrix `A` into three matrices:

```
A = U @ np.diag(s) @ Vt
```

Where:
- `U` — left singular vectors (m×m orthogonal matrix)
- `s` — singular values (diagonal, sorted descending)
- `Vt` — right singular vectors transposed (n×n orthogonal matrix)

SVD is used in: **image compression**, **PCA**, **recommendation systems**, **noise reduction**.

```python
A = np.array([[1., 2., 3.],
              [4., 5., 6.]])

U, s, Vt = np.linalg.svd(A)

print("U shape:", U.shape)    # (2, 2)
print("s shape:", s.shape)    # (2,)   → singular values
print("Vt shape:", Vt.shape)  # (3, 3)

print("Singular values:", s)  # [9.525 0.514]
```

### Reconstruct Original Matrix

```python
# Reconstruct A from U, s, Vt
S = np.zeros(A.shape)
S[:len(s), :len(s)] = np.diag(s)

A_reconstructed = U @ S @ Vt
print(np.allclose(A, A_reconstructed))   # True ✅
```

### Truncated SVD — Low-rank Approximation (Image Compression)

Keep only the top `k` singular values for a compressed approximation:

```python
A = np.random.rand(100, 100)
U, s, Vt = np.linalg.svd(A)

# Keep only top 10 singular values (out of 100)
k = 10
A_approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

compression_ratio = (k * (100 + 1 + 100)) / (100 * 100)
print(f"Compression ratio: {compression_ratio:.1%}")  # 20.1%
print(f"Reconstruction error: {np.linalg.norm(A - A_approx):.4f}")
```

---

## 12. np.linalg.lstsq() — Least Squares Solution

When a system of equations has **no exact solution** (overdetermined — more equations than unknowns), `lstsq()` finds the **best approximate solution** that minimizes the sum of squared errors.

```python
# Fitting a line y = mx + c to 4 noisy data points
# 4 equations, 2 unknowns (m and c) → overdetermined

x = np.array([0., 1., 2., 3.])
y = np.array([1.1, 2.9, 5.2, 6.8])   # noisy linear data

# Build the matrix A: each row is [x_i, 1]
A = np.column_stack([x, np.ones(len(x))])
print(A)
# [[0. 1.]
#  [1. 1.]
#  [2. 1.]
#  [3. 1.]]

# Solve for [m, c]
result = np.linalg.lstsq(A, y, rcond=None)
m, c = result[0]

print(f"Slope (m): {m:.4f}")       # ~2.0
print(f"Intercept (c): {c:.4f}")   # ~1.0
print(f"Best fit line: y = {m:.2f}x + {c:.2f}")
```

### Return Values of lstsq()

```python
solution, residuals, rank, singular_values = np.linalg.lstsq(A, y, rcond=None)
# solution  → best fit coefficients
# residuals → sum of squared errors (empty if not full rank)
# rank      → rank of matrix A
# singular_values → singular values of A
```

---

## 13. What is the Random Module?

The **random module** (`np.random`) generates **pseudo-random numbers** — numbers that appear random but are produced by a deterministic algorithm.

Random numbers are essential for:
- **Simulations** (Monte Carlo methods)
- **Machine learning** (weight initialization, dropout)
- **Statistics** (bootstrapping, hypothesis testing)
- **Games** (dice rolls, shuffling cards)
- **Data augmentation** (adding noise to training data)

### Old API vs New API

```python
# Old API (still works, but less flexible)
np.random.rand(3)

# New API (recommended since NumPy 1.17)
rng = np.random.default_rng(seed=42)
rng.random(3)
```

We will cover both, but the **new API is preferred**.

---

## 14. np.random.seed() — Reproducibility

Random numbers are generated from an internal **state**. Setting a **seed** makes the sequence of random numbers **reproducible** — the same seed always gives the same numbers.

```python
# Without seed — different every run
print(np.random.rand(3))   # e.g. [0.374 0.951 0.732]
print(np.random.rand(3))   # e.g. [0.599 0.156 0.058]  ← different!

# With seed — same every run
np.random.seed(42)
print(np.random.rand(3))   # [0.374 0.951 0.732]

np.random.seed(42)          # reset seed
print(np.random.rand(3))   # [0.374 0.951 0.732]  ← same as before!
```

### New API — Generator with seed

```python
rng = np.random.default_rng(seed=42)

print(rng.random(3))   # always the same 3 numbers for seed=42
print(rng.random(3))   # next 3 numbers in the sequence
```

### Why use seeds?

- **Reproducibility** — experiments give the same result every time
- **Debugging** — easier to reproduce bugs
- **Collaboration** — others can reproduce your exact results

---

## 15. Generating Random Numbers

### Uniform distribution — np.random.rand()

Generates floats uniformly distributed between **0 and 1**:

```python
# Old API
print(np.random.rand())        # single float in [0, 1)
print(np.random.rand(5))       # 1D array of 5 floats
print(np.random.rand(3, 4))    # 2D array shape (3,4)

# New API
rng = np.random.default_rng(42)
print(rng.random())            # single float
print(rng.random(5))           # 1D array
print(rng.random((3, 4)))      # 2D array
```

### Uniform distribution over custom range — np.random.uniform()

```python
# Random floats in [low, high)
print(np.random.uniform(10, 20, size=5))    # 5 values between 10 and 20
print(np.random.uniform(-1, 1, size=(3,3))) # 3×3 matrix between -1 and 1

# New API
rng.uniform(10, 20, size=5)
```

---

## 16. Random Integers — randint()

`np.random.randint()` generates random integers in a given range.

```python
# Old API
print(np.random.randint(1, 7))           # single int in [1, 6] — dice roll!
print(np.random.randint(0, 100, size=5)) # 5 ints in [0, 99]
print(np.random.randint(1, 10, size=(3, 3))) # 3×3 matrix

# New API
rng = np.random.default_rng(42)
print(rng.integers(1, 7))                # single int in [1, 6]
print(rng.integers(0, 100, size=5))      # 5 ints

# Simulate 10 dice rolls
dice = rng.integers(1, 7, size=10)
print(dice)   # e.g. [3 5 1 4 6 2 6 1 3 5]
```

### Counting outcomes

```python
rng = np.random.default_rng(42)
dice = rng.integers(1, 7, size=1000)

faces, counts = np.unique(dice, return_counts=True)
for face, count in zip(faces, counts):
    print(f"Face {face}: {count} times ({count/10:.1f}%)")
```

---

## 17. Random Choice — choice()

`np.random.choice()` selects random elements **from an array** or range.

```python
a = np.array([10, 20, 30, 40, 50])

# Old API
print(np.random.choice(a))             # pick 1 random element
print(np.random.choice(a, size=3))     # pick 3 with replacement
print(np.random.choice(a, size=3, replace=False))  # pick 3 WITHOUT replacement

# New API
rng = np.random.default_rng(42)
print(rng.choice(a, size=3))
print(rng.choice(a, size=3, replace=False))
```

### Weighted choice — custom probabilities

```python
items = ['apple', 'banana', 'cherry']
probs = [0.6, 0.3, 0.1]   # apple 60%, banana 30%, cherry 10%

rng = np.random.default_rng(42)
picks = rng.choice(items, size=10, p=probs)
print(picks)   # mostly apples

# Count how many times each was picked
unique, counts = np.unique(picks, return_counts=True)
for item, count in zip(unique, counts):
    print(f"{item}: {count}")
```

### Simulating a coin flip

```python
rng = np.random.default_rng(42)
flips = rng.choice(['H', 'T'], size=20)
print(flips)
print(f"Heads: {(flips=='H').sum()}, Tails: {(flips=='T').sum()}")
```

---

## 18. Random Shuffle — shuffle() and permutation()

### np.random.shuffle() — Shuffle In-Place

```python
a = np.array([1, 2, 3, 4, 5])

np.random.shuffle(a)    # modifies array in place
print(a)                # e.g. [3 1 5 2 4]
```

### np.random.permutation() — Return Shuffled Copy

```python
a = np.array([1, 2, 3, 4, 5])

b = np.random.permutation(a)   # returns new shuffled array
print(a)   # [1 2 3 4 5]  ← original unchanged
print(b)   # [3 1 5 2 4]  ← shuffled copy

# Shuffle indices 0 to n-1
print(np.random.permutation(5))   # e.g. [2 0 4 1 3]
```

### New API

```python
rng = np.random.default_rng(42)
a = np.array([1, 2, 3, 4, 5])

rng.shuffle(a)              # in-place
b = rng.permutation(a)      # copy
```

### Real-World Use — Shuffle training data

```python
rng = np.random.default_rng(42)

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([0, 1, 0, 1, 0])

# Shuffle both X and y the same way
idx = rng.permutation(len(X))
X_shuffled = X[idx]
y_shuffled = y[idx]

print("Shuffled X:", X_shuffled)
print("Shuffled y:", y_shuffled)
```

---

## 19. Probability Distributions

NumPy can sample from many statistical distributions. Here are the most commonly used:

### Normal (Gaussian) Distribution — randn() / normal()

Bell-shaped curve. Most values cluster around the mean.

```python
# Standard normal: mean=0, std=1
print(np.random.randn(5))             # 5 values from N(0,1)
print(np.random.randn(3, 3))          # 3×3 matrix

# Custom mean and std
print(np.random.normal(loc=50, scale=10, size=5))  # mean=50, std=10

# New API
rng = np.random.default_rng(42)
print(rng.normal(loc=50, scale=10, size=5))
```

### Binomial Distribution — binomial()

Number of successes in `n` independent trials, each with probability `p`.

```python
# n=10 coin flips, p=0.5 → how many heads?
results = np.random.binomial(n=10, p=0.5, size=1000)

print(f"Mean heads: {results.mean():.2f}")    # ~5.0
print(f"Std:        {results.std():.2f}")     # ~1.58
```

### Poisson Distribution — poisson()

Number of events in a fixed time period, given an average rate.

```python
# Average 3 emails per hour → simulate 10 hours
emails = np.random.poisson(lam=3, size=10)
print(emails)   # e.g. [2 4 3 5 2 1 3 4 2 3]
```

### Exponential Distribution — exponential()

Time between events in a Poisson process.

```python
# Average time between events = 2 minutes
wait_times = np.random.exponential(scale=2.0, size=5)
print(np.round(wait_times, 2))
```

### Uniform Distribution — uniform()

All values equally likely between low and high.

```python
print(np.random.uniform(low=0, high=10, size=5))
```

### Summary Table

| Distribution | Function | Use Case |
|-------------|----------|----------|
| Normal | `np.random.normal(mean, std, size)` | Heights, errors, noise |
| Standard Normal | `np.random.randn(size)` | Weights in neural networks |
| Binomial | `np.random.binomial(n, p, size)` | Coin flips, pass/fail |
| Poisson | `np.random.poisson(lam, size)` | Event counts per interval |
| Exponential | `np.random.exponential(scale, size)` | Time between events |
| Uniform | `np.random.uniform(low, high, size)` | Equal probability ranges |
| Integer | `np.random.randint(low, high, size)` | Dice, random indices |

---

## 20. Real-World Example — Simulation & Linear System

### Part A: Solving a Real Linear System

A store sells 3 products. We know:
- Revenue on day 1: product A×5 + B×3 + C×2 = $230
- Revenue on day 2: product A×2 + B×4 + C×3 = $180
- Revenue on day 3: product A×3 + B×2 + C×5 = $200

Find the price of each product:

```python
import numpy as np

# Coefficient matrix (quantities sold)
A = np.array([[5., 3., 2.],
              [2., 4., 3.],
              [3., 2., 5.]])

# Revenue vector
b = np.array([230., 180., 200.])

# Check the system has a unique solution
print(f"Determinant: {np.linalg.det(A):.2f}")
print(f"Rank: {np.linalg.matrix_rank(A)}")

# Solve for prices
prices = np.linalg.solve(A, b)
print(f"\nProduct A price: ${prices[0]:.2f}")
print(f"Product B price: ${prices[1]:.2f}")
print(f"Product C price: ${prices[2]:.2f}")

# Verify
print(f"\nVerification (should match {b}):")
print(np.round(A @ prices, 2))
```

---

### Part B: Monte Carlo Simulation — Estimating π

We can estimate π by randomly throwing darts at a unit square and counting how many land inside the circle:

```python
rng = np.random.default_rng(42)
n = 1_000_000   # number of darts

# Random (x, y) coordinates in [-1, 1]
x = rng.uniform(-1, 1, n)
y = rng.uniform(-1, 1, n)

# Distance from origin
distances = np.sqrt(x**2 + y**2)

# Count darts inside unit circle (distance ≤ 1)
inside = (distances <= 1).sum()

# Estimate π = 4 × (inside / total)
pi_estimate = 4 * inside / n
print(f"Estimated π: {pi_estimate:.5f}")
print(f"Actual π:    {np.pi:.5f}")
print(f"Error:       {abs(pi_estimate - np.pi):.5f}")
```

---

### Part C: Simulating Student Exam Scores

```python
rng = np.random.default_rng(42)

n_students = 50
n_subjects = 4
subjects   = ['Math', 'Science', 'English', 'History']

# Simulate scores: normally distributed, different mean/std per subject
means  = [72, 68, 75, 70]
stds   = [12,  10,  8, 11]

scores = np.column_stack([
    rng.normal(m, s, n_students).clip(0, 100).astype(int)
    for m, s in zip(means, stds)
])

print("Score Statistics:")
print(f"{'Subject':<10} {'Mean':>6} {'Std':>6} {'Min':>5} {'Max':>5}")
print("-" * 35)
for i, subj in enumerate(subjects):
    col = scores[:, i]
    print(f"{subj:<10} {col.mean():>6.1f} {col.std():>6.1f} {col.min():>5} {col.max():>5}")

# Correlation between subjects
corr = np.corrcoef(scores.T)
print("\nCorrelation Matrix:")
for i, subj in enumerate(subjects):
    row = [f"{corr[i,j]:.2f}" for j in range(len(subjects))]
    print(f"  {subj:<10}: {' '.join(row)}")
```

---

## 21. Practice Exercises

### Exercise 1 — Matrix Operations
```python
A = np.array([[2., 1., 0.],
              [1., 3., 1.],
              [0., 1., 2.]])
b = np.array([4., 9., 6.])

# Q1: Calculate the determinant of A
# Q2: Solve the linear system A @ x = b
# Q3: Verify your solution
# Q4: Find the eigenvalues of A
```

<details>
<summary>Show Answers</summary>

```python
print(np.linalg.det(A))          # non-zero → has solution

x = np.linalg.solve(A, b)
print("Solution:", np.round(x, 4))

print("Verify:", np.allclose(A @ x, b))

vals, vecs = np.linalg.eig(A)
print("Eigenvalues:", np.round(vals, 4))
```
</details>

---

### Exercise 2 — Norms & Distances
```python
cities = np.array([[0., 0.],    # City A (origin)
                   [3., 4.],    # City B
                   [1., 7.],    # City C
                   [6., 2.]])   # City D

# Q1: Distance from city A to each other city
# Q2: Which city is closest to city A?
# Q3: Normalize the coordinates of city B to a unit vector
```

<details>
<summary>Show Answers</summary>

```python
A_pos = cities[0]
distances = np.array([np.linalg.norm(city - A_pos) for city in cities[1:]])
print("Distances:", distances)

closest = ['B', 'C', 'D'][distances.argmin()]
print(f"Closest city: {closest}")

B = cities[1]
B_unit = B / np.linalg.norm(B)
print("Unit vector:", B_unit, "| norm:", np.linalg.norm(B_unit))
```
</details>

---

### Exercise 3 — Random Module
```python
rng = np.random.default_rng(99)

# Q1: Simulate rolling 2 dice 100 times — find the most common sum
# Q2: Generate 1000 samples from N(mean=65, std=15) — what % are above 80?
# Q3: Randomly select 5 unique students from a class of 20 (no replacement)
```

<details>
<summary>Show Answers</summary>

```python
dice1 = rng.integers(1, 7, 100)
dice2 = rng.integers(1, 7, 100)
sums  = dice1 + dice2
vals, counts = np.unique(sums, return_counts=True)
print(f"Most common sum: {vals[counts.argmax()]}")

heights = rng.normal(65, 15, 1000)
print(f"Above 80: {(heights > 80).mean() * 100:.1f}%")

students = np.arange(1, 21)
selected = rng.choice(students, size=5, replace=False)
print(f"Selected: {selected}")
```
</details>

---

### Exercise 4 — Simulation
```python
rng = np.random.default_rng(42)
# Q1: Estimate the probability that the sum of 3 dice ≥ 15
#     using 100,000 simulations
# Q2: Generate a 5×5 random matrix and find its SVD
#     then reconstruct it and verify
```

<details>
<summary>Show Answers</summary>

```python
dice = rng.integers(1, 7, size=(100_000, 3))
prob = (dice.sum(axis=1) >= 15).mean()
print(f"P(sum ≥ 15) ≈ {prob:.4f}")

A = rng.random((5, 5))
U, s, Vt = np.linalg.svd(A)
S = np.zeros_like(A, dtype=float)
S[:5, :5] = np.diag(s)
A_reconstructed = U @ S @ Vt
print(f"Reconstruction match: {np.allclose(A, A_reconstructed)}")
```
</details>

---

## 22. Cheat Sheet

```python
import numpy as np

# ── SPECIAL MATRICES ──────────────────────────────────────
np.eye(n)              # n×n identity matrix
np.zeros((m, n))       # zero matrix
np.diag([1, 2, 3])     # diagonal matrix from values
np.diag(A)             # extract diagonal from matrix A
np.triu(A)             # upper triangular
np.tril(A)             # lower triangular

# ── MATRIX OPERATIONS ─────────────────────────────────────
A @ B                  # matrix multiplication (preferred)
np.matmul(A, B)        # same as A @ B
np.dot(A, B)           # dot product (works for 1D and 2D)
A.T                    # transpose
A * B                  # element-wise multiplication (NOT matmul!)

# ── np.linalg ─────────────────────────────────────────────
np.linalg.inv(A)             # matrix inverse
np.linalg.det(A)             # determinant
np.linalg.solve(A, b)        # solve A @ x = b
np.linalg.eig(A)             # eigenvalues and eigenvectors
np.linalg.eigvals(A)         # eigenvalues only (faster)
np.linalg.eigh(A)            # for symmetric matrices
np.linalg.norm(v)            # L2 norm (default)
np.linalg.norm(v, ord=1)     # L1 norm
np.linalg.norm(v, ord=np.inf) # L-infinity norm
np.linalg.matrix_rank(A)     # rank of matrix
np.linalg.svd(A)             # singular value decomposition
np.linalg.lstsq(A, b, rcond=None) # least squares solution
np.allclose(A, B)            # check if two arrays are close

# ── RANDOM — OLD API ──────────────────────────────────────
np.random.seed(42)              # set seed for reproducibility
np.random.rand(m, n)            # uniform [0, 1)
np.random.randn(m, n)           # standard normal N(0, 1)
np.random.randint(low, high, size)  # random integers
np.random.normal(mean, std, size)   # normal distribution
np.random.uniform(low, high, size)  # uniform distribution
np.random.binomial(n, p, size)      # binomial distribution
np.random.poisson(lam, size)        # Poisson distribution
np.random.choice(a, size, replace)  # random selection
np.random.shuffle(a)                # shuffle in-place
np.random.permutation(a)            # shuffled copy

# ── RANDOM — NEW API (preferred) ──────────────────────────
rng = np.random.default_rng(seed=42)
rng.random(size)               # uniform [0, 1)
rng.normal(mean, std, size)    # normal distribution
rng.integers(low, high, size)  # random integers
rng.uniform(low, high, size)   # uniform distribution
rng.choice(a, size, replace, p)# random selection with weights
rng.shuffle(a)                 # in-place shuffle
rng.permutation(a)             # shuffled copy
```

---

## 🔗 What's Next?

After mastering Day 5, you're ready for:

➡️ **Day 6 — I/O, Structured Arrays & Performance**
Learn how to save/load arrays to disk, work with structured data types, understand memory layout, and avoid subtle memory bugs with views vs copies.

---

## 📚 Resources

- [NumPy Linear Algebra Docs](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [NumPy Random Docs](https://numpy.org/doc/stable/reference/random/index.html)
- [NumPy Random Generator](https://numpy.org/doc/stable/reference/random/generator.html)
- [Practice on Google Colab](https://colab.research.google.com/)

---

*Part of the [7-Day NumPy Learning Plan](./README.md) · Day 5 of 7*
