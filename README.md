# 📘 7-Day NumPy Learning Plan

A structured, beginner-to-intermediate roadmap for mastering NumPy — the core Python library for numerical computing.

---

## 🧰 Prerequisites

- Basic Python knowledge (lists, loops, functions, conditionals)
- Python 3.7+ installed
- Recommended tools: [Jupyter Notebook](https://jupyter.org/) or [Google Colab](https://colab.research.google.com/) (no setup required)

### Install NumPy

```bash
pip install numpy
```

---

## 📅 The Plan

### Day 1 — NumPy Basics & Array Creation

**Topics:**
- `np.array`, `np.zeros`, `np.ones`, `np.arange`, `np.linspace`
- Data types (`dtype`)
- Array attributes: `ndim`, `shape`, `size`

**Goal:** Create arrays confidently and inspect their properties.

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.zeros((3, 3))
c = np.arange(0, 10, 2)
print(a.shape, b.dtype, c.ndim)
```

---

### Day 2 — Indexing, Slicing & Reshaping

**Topics:**
- 1D / 2D / 3D slicing
- Boolean indexing
- Fancy indexing
- `reshape`, `flatten`, `ravel`, `transpose`

**Goal:** Extract and rearrange any part of an array.

```python
a = np.arange(12).reshape(3, 4)
print(a[1, :])          # row slice
print(a[a > 5])         # boolean mask
print(a.T)              # transpose
```

---

### Day 3 — Broadcasting & Vectorized Operations

**Topics:**
- Element-wise arithmetic on arrays
- Broadcasting rules
- Universal functions (ufuncs): `np.sqrt`, `np.exp`, `np.log`
- Avoiding Python loops

**Goal:** Write fast, loop-free numerical code.

```python
a = np.array([1, 2, 3])
b = np.array([[10], [20], [30]])
print(a + b)            # broadcasting
print(np.sqrt(a))       # ufunc
```

---

### Day 4 — Aggregation & Statistics

**Topics:**
- `sum`, `mean`, `std`, `min`, `max` with `axis` parameter
- `cumsum`, `diff`
- `np.where`, `np.unique`, `np.sort`

**Goal:** Summarize and analyze datasets along any axis.

```python
a = np.array([[1, 2], [3, 4]])
print(a.mean(axis=0))       # column means
print(np.where(a > 2, 1, 0))
```

---

### Day 5 — Linear Algebra & Random Module

**Topics:**
- `np.linalg`: `dot`, `matmul`, `inv`, `det`, `eig`
- `np.random`: `rand`, `randn`, `randint`, `seed`, `choice`

**Goal:** Solve matrix problems and simulate data.

```python
A = np.array([[1, 2], [3, 4]])
print(np.linalg.det(A))
print(np.linalg.inv(A))

rng = np.random.default_rng(seed=42)
print(rng.normal(0, 1, size=(3, 3)))
```

---

### Day 6 — I/O, Structured Arrays & Performance

**Topics:**
- `np.save` / `np.load`, `np.savetxt` / `np.loadtxt`
- Structured (record) dtypes
- Views vs. copies
- Memory layout: C-order vs. Fortran-order

**Goal:** Read/write data efficiently and avoid memory bugs.

```python
a = np.array([1.0, 2.0, 3.0])
np.save('my_array.npy', a)
b = np.load('my_array.npy')

# View vs copy
view = a[:]        # shares memory
copy = a.copy()    # independent
```

---

### Day 7 — Capstone: Real Data Mini-Project

**Topics:**
- Load a CSV dataset with `np.loadtxt` or `np.genfromtxt`
- Clean data using boolean masks
- Compute descriptive statistics
- Visualize results with Matplotlib

**Goal:** Apply all 6 days to a real-world problem.

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
clean = data[~np.isnan(data).any(axis=1)]  # remove NaN rows

print("Mean:", clean.mean(axis=0))
print("Std:", clean.std(axis=0))

plt.hist(clean[:, 0], bins=20)
plt.title("Feature Distribution")
plt.show()
```

---

## 🗓️ Daily Schedule

| Day | Topic | Time Estimate |
|-----|-------|---------------|
| 1 | Basics & array creation | 1–1.5 hrs |
| 2 | Indexing, slicing & reshaping | 1–1.5 hrs |
| 3 | Broadcasting & vectorized ops | 1.5–2 hrs |
| 4 | Aggregation & statistics | 1–1.5 hrs |
| 5 | Linear algebra & random | 1.5–2 hrs |
| 6 | I/O & performance | 1–1.5 hrs |
| 7 | Capstone mini-project | 2–3 hrs |

---

## 📚 Resources

- [NumPy Official Docs](https://numpy.org/doc/)
- [NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [100 NumPy Exercises](https://github.com/rougier/numpy-100)
- [Google Colab](https://colab.research.google.com/) — free browser-based notebook

---

## 💡 Tips

- Always practice in a notebook — reading alone won't stick.
- Day 3 (broadcasting) is the hardest concept. Take your time and experiment.
- After Day 7, explore **Pandas** (data analysis) and **SciPy** (scientific computing) as next steps.

---

*Happy learning! 🚀*
