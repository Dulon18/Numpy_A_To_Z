# 📘 Day 4 — NumPy Aggregation & Statistics (A to Z Guide)

A complete, beginner-friendly deep dive into summarizing, analyzing, and extracting insights from arrays using NumPy.

---

## 📋 Table of Contents

1. [What is Aggregation?](#1-what-is-aggregation)
2. [Understanding the axis Parameter](#2-understanding-the-axis-parameter)
3. [sum() — Adding Elements](#3-sum--adding-elements)
4. [mean() — Average Value](#4-mean--average-value)
5. [std() and var() — Spread of Data](#5-std-and-var--spread-of-data)
6. [min() and max() — Extreme Values](#6-min-and-max--extreme-values)
7. [argmin() and argmax() — Index of Extremes](#7-argmin-and-argmax--index-of-extremes)
8. [median() and percentile()](#8-median-and-percentile)
9. [cumsum() — Cumulative Sum](#9-cumsum--cumulative-sum)
10. [diff() — Differences Between Elements](#10-diff--differences-between-elements)
11. [np.sort() and np.argsort()](#11-npsort-and-npargsort)
12. [np.unique() — Finding Unique Values](#12-npunique--finding-unique-values)
13. [np.where() — Advanced Filtering](#13-npwhere--advanced-filtering)
14. [keepdims — Preserving Shape](#14-keepdims--preserving-shape)
15. [np.corrcoef() and np.cov() — Correlation & Covariance](#15-npcorrcoef-and-npcov--correlation--covariance)
16. [np.histogram() — Frequency Distribution](#16-nphistogram--frequency-distribution)
17. [Real-World Example — Student Score Analysis](#17-real-world-example--student-score-analysis)
18. [Practice Exercises](#18-practice-exercises)
19. [Cheat Sheet](#19-cheat-sheet)

---

## 1. What is Aggregation?

**Aggregation** means reducing many values down to a **single summary value** (or fewer values).

For example:
- 100 exam scores → 1 average score
- A month of daily temperatures → 1 maximum temperature
- Sales data for 12 months → 1 total annual revenue

```python
import numpy as np

scores = np.array([78, 92, 85, 60, 95, 73, 88])

print(scores.sum())    # 571   → total
print(scores.mean())   # 81.57 → average
print(scores.max())    # 95    → highest
print(scores.min())    # 60    → lowest
print(scores.std())    # 11.56 → spread
```

NumPy aggregation functions are:
- **Extremely fast** — implemented in C
- **Flexible** — work on any dimension via the `axis` parameter
- **Concise** — replace many lines of manual code

---

## 2. Understanding the axis Parameter

The `axis` parameter is the **most important concept** in NumPy aggregation. It controls **which direction** the aggregation happens.

### For a 2D Array

```
         axis=1 →→→ (across columns)
         ┌────┬────┬────┐
axis  ↓  │  1 │  2 │  3 │  → row 0
  =   ↓  ├────┼────┼────┤
  0   ↓  │  4 │  5 │  6 │  → row 1
         ├────┼────┼────┤
         │  7 │  8 │  9 │  → row 2
         └────┴────┴────┘
```

- **`axis=0`** → operates **down** each column → result has shape `(cols,)`
- **`axis=1`** → operates **across** each row → result has shape `(rows,)`
- **No axis** → operates on **all elements** → result is a scalar

```python
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(a.sum())          # 45         → all elements
print(a.sum(axis=0))    # [12 15 18] → column sums (down ↓)
print(a.sum(axis=1))    # [ 6 15 24] → row sums (across →)
```

### Visual Explanation

```
axis=0 (collapse rows → result per column):
[1+4+7, 2+5+8, 3+6+9] = [12, 15, 18]

axis=1 (collapse columns → result per row):
[1+2+3, 4+5+6, 7+8+9] = [6, 15, 24]
```

### Memory Tip 🧠

> **"axis=0 collapses rows, axis=1 collapses columns"**
> The axis you specify is the one that **disappears**.

---

## 3. sum() — Adding Elements

`sum()` adds up elements along a given axis.

```python
a = np.array([[10, 20, 30],
              [40, 50, 60]])

# Total sum
print(a.sum())           # 210

# Sum of each column (axis=0 → rows collapse)
print(a.sum(axis=0))     # [ 50  70  90]

# Sum of each row (axis=1 → columns collapse)
print(a.sum(axis=1))     # [ 60 150]
```

### Counting True values with sum()

Since `True = 1` and `False = 0`, `sum()` can count how many elements meet a condition:

```python
a = np.array([10, 25, 3, 47, 8, 60, 15])

count_above_20 = (a > 20).sum()
print(count_above_20)   # 3   → [25, 47, 60]

count_even = (a % 2 == 0).sum()
print(count_even)       # 4   → [10, 8, 60] wait...
# [10→even, 25→odd, 3→odd, 47→odd, 8→even, 60→even, 15→odd] = 3
```

### np.sum() vs Python sum()

```python
a = np.array([1, 2, 3, 4, 5])

print(np.sum(a))   # ✅ Fast — NumPy C implementation
print(sum(a))      # ⚠️  Slower — Python built-in, no axis support
```

Always use `np.sum()` or `a.sum()` for NumPy arrays.

---

## 4. mean() — Average Value

`mean()` calculates the **arithmetic average**: sum divided by count.

```python
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(a.mean())          # 5.0   → overall average
print(a.mean(axis=0))    # [4. 5. 6.]  → column averages
print(a.mean(axis=1))    # [2. 5. 8.]  → row averages
```

### Weighted Average with np.average()

`np.average()` allows weights — useful in grading systems or surveys:

```python
scores     = np.array([80, 90, 70, 85])
weights    = np.array([0.2, 0.3, 0.1, 0.4])   # must sum to 1 (or NumPy normalizes)

weighted_avg = np.average(scores, weights=weights)
print(weighted_avg)   # 83.5
# = 80×0.2 + 90×0.3 + 70×0.1 + 85×0.4
```

### np.nanmean() — Ignore NaN values

Real-world data often has missing values (`NaN`). Use `np.nanmean()` to skip them:

```python
a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

print(a.mean())         # nan      ← NaN "poisons" the result
print(np.nanmean(a))    # 3.0      ← ignores NaN values
```

---

## 5. std() and var() — Spread of Data

**Standard deviation** (`std`) and **variance** (`var`) measure how **spread out** values are from the mean.

- **Low std** = values are clustered close to the mean
- **High std** = values are spread far from the mean

### Formula

```
variance   = mean( (x - mean)² )
std        = sqrt( variance )
```

```python
a = np.array([2, 4, 4, 4, 5, 5, 7, 9])

print(a.mean())   # 5.0
print(a.var())    # 4.0
print(a.std())    # 2.0
```

### Example on 2D Array

```python
data = np.array([[10, 20, 30],
                 [40, 50, 60]])

print(data.std())           # 18.0 → overall std
print(data.std(axis=0))     # [15. 15. 15.] → std per column
print(data.std(axis=1))     # [8.165 8.165]  → std per row
```

### ddof — Population vs Sample std

By default, NumPy calculates **population std** (divides by N).
For **sample std** (divides by N-1), use `ddof=1`:

```python
a = np.array([2, 4, 4, 4, 5, 5, 7, 9])

print(a.std())          # 2.0  → population std  (ddof=0)
print(a.std(ddof=1))    # 2.138 → sample std      (ddof=1)
```

Use `ddof=1` when your data is a **sample** from a larger population (common in statistics).

---

## 6. min() and max() — Extreme Values

`min()` and `max()` find the smallest and largest values.

```python
a = np.array([[3, 7, 1],
              [9, 2, 8],
              [4, 6, 5]])

print(a.min())           # 1   → overall minimum
print(a.max())           # 9   → overall maximum

print(a.min(axis=0))     # [3 2 1]  → minimum in each column
print(a.max(axis=0))     # [9 7 8]  → maximum in each column

print(a.min(axis=1))     # [1 2 4]  → minimum in each row
print(a.max(axis=1))     # [7 9 6]  → maximum in each row
```

### np.nanmin() / np.nanmax() — Ignore NaN

```python
a = np.array([3.0, np.nan, 1.0, 7.0, np.nan])

print(np.nanmin(a))   # 1.0
print(np.nanmax(a))   # 7.0
```

### Peak-to-Peak Range (ptp)

```python
a = np.array([10, 25, 3, 47, 8])

print(np.ptp(a))   # 44   → max - min = 47 - 3
```

---

## 7. argmin() and argmax() — Index of Extremes

`argmin()` and `argmax()` return the **index** of the minimum/maximum value — not the value itself.

This is extremely useful when you need to **find which item** is largest/smallest, not just what value it has.

```python
a = np.array([30, 10, 50, 20, 40])

print(a.argmin())   # 1   → index of smallest value (10)
print(a.argmax())   # 2   → index of largest value (50)

# Use the index to get the value back
print(a[a.argmax()])   # 50
```

### On 2D Arrays

```python
a = np.array([[3, 7, 1],
              [9, 2, 8]])

print(a.argmax())          # 3   → flat index (9 is at position 3)
print(a.argmax(axis=0))    # [1 0 1]  → row index of max in each column
print(a.argmax(axis=1))    # [1 0]    → col index of max in each row
```

### Real-World Use Case

```python
products  = ['Apple', 'Banana', 'Cherry', 'Date']
sales     = np.array([320, 150, 480, 90])

best_seller_idx  = sales.argmax()
worst_seller_idx = sales.argmin()

print(f"Best seller:  {products[best_seller_idx]}")   # Cherry
print(f"Worst seller: {products[worst_seller_idx]}")  # Date
```

---

## 8. median() and percentile()

### np.median() — Middle Value

The **median** is the middle value when data is sorted. It's more robust than mean when data has outliers.

```python
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])

print(np.mean(a))     # 3.875
print(np.median(a))   # 3.5    → middle value after sorting: [1,1,2,3,4,5,6,9]
```

### Median vs Mean with Outliers

```python
salaries = np.array([30000, 32000, 35000, 31000, 500000])  # one CEO salary

print(np.mean(salaries))     # 125600  ← skewed by outlier
print(np.median(salaries))   # 32000   ← not affected by outlier
```

The **median is better** when data has extreme outliers.

### np.percentile() — Divide Data into Percentiles

A **percentile** tells you what percentage of values fall below a given point.

```python
scores = np.array([55, 60, 65, 70, 75, 80, 85, 90, 95, 100])

print(np.percentile(scores, 25))    # 66.25  → 25th percentile (Q1)
print(np.percentile(scores, 50))    # 77.5   → 50th percentile (median)
print(np.percentile(scores, 75))    # 88.75  → 75th percentile (Q3)
print(np.percentile(scores, 90))    # 95.5   → 90th percentile
```

### Interquartile Range (IQR)

```python
Q1  = np.percentile(scores, 25)
Q3  = np.percentile(scores, 75)
IQR = Q3 - Q1
print(IQR)   # 22.5

# Detect outliers: values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = scores[(scores < lower) | (scores > upper)]
print(outliers)   # []  → no outliers in this dataset
```

### np.quantile() — Same as percentile but uses 0–1 scale

```python
print(np.quantile(scores, 0.25))   # 66.25  → same as percentile 25
print(np.quantile(scores, 0.75))   # 88.75  → same as percentile 75
```

---

## 9. cumsum() — Cumulative Sum

`cumsum()` returns a **running total** — each element is the sum of all previous elements plus itself.

```python
a = np.array([1, 2, 3, 4, 5])

print(a.cumsum())   # [ 1  3  6 10 15]
# 1, 1+2=3, 3+3=6, 6+4=10, 10+5=15
```

### 2D cumsum()

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.cumsum())           # [ 1  3  6 10 15 21]  → flattened running total
print(a.cumsum(axis=0))     # running total down each column
# [[ 1  2  3]
#  [ 5  7  9]]

print(a.cumsum(axis=1))     # running total across each row
# [[ 1  3  6]
#  [ 4  9 15]]
```

### Real-World Use Case — Running total of sales

```python
monthly_sales = np.array([1500, 2300, 1800, 2700, 3100, 2500])
running_total = monthly_sales.cumsum()

for month, total in enumerate(running_total, 1):
    print(f"Month {month}: cumulative sales = ${total:,}")

# Month 1: cumulative sales = $1,500
# Month 2: cumulative sales = $3,800
# Month 3: cumulative sales = $5,600
# ...
```

### cumprod() — Cumulative Product

```python
a = np.array([1, 2, 3, 4, 5])
print(a.cumprod())   # [  1   2   6  24 120]
# 1, 1×2=2, 2×3=6, 6×4=24, 24×5=120
```

---

## 10. diff() — Differences Between Elements

`np.diff()` computes the **difference between consecutive elements**. Great for finding changes, trends, or rates.

```python
a = np.array([10, 15, 21, 28, 32, 45])

print(np.diff(a))   # [ 5  6  7  4 13]
# 15-10=5, 21-15=6, 28-21=7, 32-28=4, 45-32=13
```

Note: the result has **one fewer element** than the original.

### Higher-order differences (n parameter)

```python
a = np.array([1, 4, 9, 16, 25])   # perfect squares

print(np.diff(a))      # [ 3  5  7  9]    → 1st differences
print(np.diff(a, n=2)) # [2 2 2]          → 2nd differences (constant → quadratic sequence)
```

### Real-World Use Case — Daily stock price changes

```python
prices    = np.array([100, 105, 102, 108, 115, 112, 120])
changes   = np.diff(prices)

print("Daily change:", changes)        # [  5  -3   6   7  -3   8]
print("Up days:", (changes > 0).sum()) # 4
print("Down days:", (changes < 0).sum()) # 2

# Percentage change
pct_change = np.diff(prices) / prices[:-1] * 100
print("% change:", np.round(pct_change, 2))
```

### 2D diff()

```python
a = np.array([[1, 3, 6],
              [2, 5, 9]])

print(np.diff(a, axis=1))   # diff across columns
# [[2 3]
#  [3 4]]
```

---

## 11. np.sort() and np.argsort()

### np.sort() — Sort Array Values

```python
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])

print(np.sort(a))        # [1 1 2 3 4 5 6 9]  → ascending (default)
print(np.sort(a)[::-1])  # [9 6 5 4 3 2 1 1]  → descending (reverse the result)
```

### sort() on 2D Arrays

```python
a = np.array([[3, 1, 4],
              [9, 2, 7],
              [5, 8, 6]])

print(np.sort(a, axis=0))   # sort each column
# [[3 1 4]
#  [5 2 6]
#  [9 8 7]]

print(np.sort(a, axis=1))   # sort each row
# [[1 3 4]
#  [2 7 9]
#  [5 6 8]]
```

### ⚠️ np.sort() vs a.sort()

```python
a = np.array([3, 1, 4, 1, 5])

b = np.sort(a)   # Returns a SORTED COPY, original unchanged
print(a)         # [3 1 4 1 5]   ← unchanged
print(b)         # [1 1 3 4 5]

a.sort()         # Sorts IN PLACE, modifies original
print(a)         # [1 1 3 4 5]   ← changed!
```

### np.argsort() — Indices That Would Sort the Array

`argsort()` returns the **indices** that would sort the array — very useful for reordering related data.

```python
a = np.array([30, 10, 50, 20, 40])

idx = np.argsort(a)
print(idx)       # [1 3 0 4 2]
# a[1]=10 is smallest, a[3]=20 is 2nd, a[0]=30 is 3rd...

print(a[idx])    # [10 20 30 40 50]  → sorted values via indices
```

### Real-World Use — Sort students by score

```python
students = np.array(['Alice', 'Bob', 'Carol', 'Dave'])
scores   = np.array([85, 92, 78, 95])

rank_idx  = np.argsort(scores)[::-1]   # descending order
print("Ranking:")
for rank, i in enumerate(rank_idx, 1):
    print(f"  {rank}. {students[i]}: {scores[i]}")

# 1. Dave: 95
# 2. Bob: 92
# 3. Alice: 85
# 4. Carol: 78
```

---

## 12. np.unique() — Finding Unique Values

`np.unique()` returns the **sorted unique values** from an array and optionally their counts and positions.

```python
a = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])

print(np.unique(a))   # [1 2 3 4 5 6 9]  → sorted unique values
```

### return_counts — How many times each value appears

```python
values, counts = np.unique(a, return_counts=True)

print(values)   # [1 2 3 4 5 6 9]
print(counts)   # [2 1 2 1 3 1 1]
# 1 appears 2 times, 5 appears 3 times, etc.

# Most frequent value
most_common_idx = counts.argmax()
print(f"Most common: {values[most_common_idx]} (appears {counts[most_common_idx]} times)")
# Most common: 5 (appears 3 times)
```

### return_index — First occurrence position

```python
values, indices = np.unique(a, return_index=True)
print(indices)   # position of first occurrence of each unique value
```

### return_inverse — Map back to original

```python
values, inverse = np.unique(a, return_inverse=True)
print(inverse)           # index into 'values' for each element in 'a'
print(values[inverse])   # reconstructs original array
```

### 2D unique()

```python
a = np.array([[1, 2], [3, 4], [1, 2], [5, 6]])
print(np.unique(a))          # [1 2 3 4 5 6]     → unique across all elements
print(np.unique(a, axis=0))  # unique rows
# [[1 2]
#  [3 4]
#  [5 6]]
```

---

## 13. np.where() — Advanced Filtering

We covered `np.where()` on Day 3, but it's also a powerful **aggregation helper** — finding positions of values that meet conditions.

```python
a = np.array([10, -5, 30, -2, 15, -8, 20])

# Get indices where values are negative
neg_indices = np.where(a < 0)
print(neg_indices)      # (array([1, 3, 5]),)
print(a[neg_indices])   # [-5 -2 -8]

# Count negatives
print(len(neg_indices[0]))   # 3

# Replace negatives with 0 (clamp)
cleaned = np.where(a < 0, 0, a)
print(cleaned)   # [10  0 30  0 15  0 20]
```

### Combining with argmax/argmin

```python
a = np.array([[5, 3, 8],
              [1, 9, 2]])

# Find row and column of the maximum value
flat_idx = a.argmax()
row, col = np.unravel_index(flat_idx, a.shape)
print(f"Max value {a[row, col]} is at row={row}, col={col}")
# Max value 9 is at row=1, col=1
```

---

## 14. keepdims — Preserving Shape

When you aggregate along an axis, the result normally **loses that dimension**. `keepdims=True` **preserves** the original number of dimensions.

### Why it matters

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

row_sum = a.sum(axis=1)
print(row_sum.shape)   # (2,)  ← 1D, hard to broadcast back

row_sum_kd = a.sum(axis=1, keepdims=True)
print(row_sum_kd.shape)   # (2, 1)  ← 2D, easy to broadcast
```

### Real-World Use — Row normalization

```python
data = np.array([[10, 20, 30],
                 [40, 50, 60],
                 [70, 80, 90]])

# Normalize each row to sum to 1 (probability distribution)
row_sums  = data.sum(axis=1, keepdims=True)   # shape (3, 1)
normalized = data / row_sums                   # broadcasting works!

print(normalized)
# [[0.167 0.333 0.5  ]
#  [0.267 0.333 0.4  ]
#  [0.292 0.333 0.375]]

print(normalized.sum(axis=1))   # [1. 1. 1.]  ← each row sums to 1 ✅
```

### Without keepdims — Broadcasting fails

```python
row_sums_no_kd = data.sum(axis=1)   # shape (3,)
data / row_sums_no_kd               # ❌ Broadcasting error!
```

---

## 15. np.corrcoef() and np.cov() — Correlation & Covariance

### np.cov() — Covariance

Covariance measures whether two variables **move together**:
- **Positive** → both increase together
- **Negative** → one increases as the other decreases
- **Near zero** → no linear relationship

```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 6])

cov_matrix = np.cov(x, y)
print(cov_matrix)
# [[2.5  2. ]
#  [2.   2.5]]

# cov(x, y) = 2.0  → positive, move together
```

### np.corrcoef() — Pearson Correlation Coefficient

Correlation **normalizes** covariance to a range of **-1 to +1**:
- **+1** = perfect positive correlation
- **-1** = perfect negative correlation
- **0** = no linear correlation

```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 6])
z = np.array([5, 4, 3, 2, 1])   # opposite of x

print(np.corrcoef(x, y))
# [[1.    0.8]
#  [0.8   1. ]]
# → x and y have correlation 0.8 (strong positive)

print(np.corrcoef(x, z))
# [[1.  -1.]
#  [-1.  1.]]
# → x and z have correlation -1 (perfect negative)
```

### Real-World Use — Feature correlation in a dataset

```python
# 5 students: [study_hours, sleep_hours, exam_score]
data = np.array([[6, 7, 85],
                 [8, 6, 92],
                 [4, 8, 70],
                 [9, 5, 95],
                 [5, 7, 75]])

study = data[:, 0]
sleep = data[:, 1]
score = data[:, 2]

print("Study vs Score correlation:", np.corrcoef(study, score)[0, 1])
# 0.996  → very strong positive correlation!

print("Sleep vs Score correlation:", np.corrcoef(sleep, score)[0, 1])
# -0.949 → strong negative correlation (more sleep, lower score? unusual data!)
```

---

## 16. np.histogram() — Frequency Distribution

`np.histogram()` counts how many values fall into each **bin (range)**. It's the foundation of histogram charts.

```python
a = np.array([1, 2, 1, 3, 5, 4, 2, 3, 3, 4, 5, 5, 5])

counts, bin_edges = np.histogram(a, bins=5)

print("Counts:", counts)      # [2 2 3 2 4]
print("Edges:", bin_edges)    # [1.  1.8  2.6  3.4  4.2  5. ]
```

### Custom bin edges

```python
scores = np.array([45, 72, 88, 56, 91, 63, 79, 82, 95, 68])

counts, edges = np.histogram(scores, bins=[0, 60, 70, 80, 90, 100])
print("F (<60):", counts[0])   # 1
print("D (60-70):", counts[1]) # 3
print("C (70-80):", counts[2]) # 2
print("B (80-90):", counts[3]) # 2
print("A (90-100):", counts[4]) # 2
```

### np.digitize() — Which bin does each element belong to?

```python
grades  = np.array([45, 72, 88, 56, 91, 63])
bins    = [0, 60, 70, 80, 90, 100]
labels  = ['F', 'D', 'C', 'B', 'A']

bin_idx = np.digitize(grades, bins) - 1
for score, idx in zip(grades, bin_idx):
    print(f"Score {score} → Grade {labels[min(idx, 4)]}")
```

---

## 17. Real-World Example — Student Score Analysis

Let's put everything together with a full data analysis example.

```python
import numpy as np

# 20 student scores across 4 subjects: Math, Science, English, History
scores = np.array([
    [85, 90, 78, 88],
    [72, 65, 80, 70],
    [95, 92, 88, 96],
    [60, 55, 70, 62],
    [88, 85, 92, 84],
    [74, 80, 75, 78],
    [91, 88, 85, 90],
    [66, 70, 68, 64],
    [79, 83, 77, 81],
    [84, 87, 82, 86],
])

subjects = ['Math', 'Science', 'English', 'History']

print("=" * 40)
print("        STUDENT SCORE ANALYSIS")
print("=" * 40)

# 1. Overall stats
print(f"\nTotal students: {scores.shape[0]}")
print(f"Subjects:       {scores.shape[1]}")

# 2. Subject averages
subject_means = scores.mean(axis=0)
print("\nSubject Averages:")
for subj, avg in zip(subjects, subject_means):
    print(f"  {subj:<10}: {avg:.1f}")

# 3. Best and worst subject
best_subj  = subjects[subject_means.argmax()]
worst_subj = subjects[subject_means.argmin()]
print(f"\nBest subject:  {best_subj}  ({subject_means.max():.1f})")
print(f"Worst subject: {worst_subj} ({subject_means.min():.1f})")

# 4. Student averages
student_avgs = scores.mean(axis=1)
top_student    = student_avgs.argmax() + 1
bottom_student = student_avgs.argmin() + 1
print(f"\nTop student:    Student {top_student} ({student_avgs.max():.1f})")
print(f"Bottom student: Student {bottom_student} ({student_avgs.min():.1f})")

# 5. Grade distribution (all scores combined)
all_scores = scores.flatten()
counts, _ = np.histogram(all_scores, bins=[0, 60, 70, 80, 90, 100])
print(f"\nGrade Distribution:")
print(f"  F (<60) : {counts[0]} scores")
print(f"  D (60s) : {counts[1]} scores")
print(f"  C (70s) : {counts[2]} scores")
print(f"  B (80s) : {counts[3]} scores")
print(f"  A (90s) : {counts[4]} scores")

# 6. Percentiles
print(f"\nScore Percentiles:")
for p in [25, 50, 75, 90]:
    print(f"  {p}th percentile: {np.percentile(all_scores, p):.1f}")

# 7. Standard deviation per subject
print(f"\nScore Variability (std):")
for subj, std in zip(subjects, scores.std(axis=0)):
    print(f"  {subj:<10}: {std:.2f}")
```

---

## 18. Practice Exercises

### Exercise 1 — Basic Aggregation
```python
sales = np.array([[120, 85, 200, 150],
                  [90,  110, 180, 130],
                  [200, 95,  220, 170]])
subjects = ['Q1', 'Q2', 'Q3', 'Q4']

# Q1: Total sales per quarter (column sums)
# Q2: Average sales per region (row means)
# Q3: Which quarter had the highest total sales?
# Q4: How many quarterly figures exceeded 150?
```

<details>
<summary>Show Answers</summary>

```python
print(sales.sum(axis=0))                          # Q totals
print(sales.mean(axis=1))                         # region averages
best_q = subjects[sales.sum(axis=0).argmax()]
print(f"Best quarter: {best_q}")                  # Q3
print((sales > 150).sum())                        # 6
```
</details>

---

### Exercise 2 — Sorting & Ranking
```python
players = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve']
scores  = np.array([740, 980, 650, 870, 920])

# Q1: Sort players by score (highest first)
# Q2: What is Alice's rank?
# Q3: Print each player with their rank and score
```

<details>
<summary>Show Answers</summary>

```python
rank_idx = np.argsort(scores)[::-1]
sorted_players = [players[i] for i in rank_idx]
print(sorted_players)

alice_idx  = 0
alice_rank = np.where(rank_idx == alice_idx)[0][0] + 1
print(f"Alice's rank: {alice_rank}")

for rank, i in enumerate(rank_idx, 1):
    print(f"{rank}. {players[i]}: {scores[i]}")
```
</details>

---

### Exercise 3 — Statistics
```python
temps = np.array([22, 25, 19, 28, 31, 17, 24, 29, 26, 20,
                  33, 18, 27, 30, 23, 21, 28, 25, 22, 26])

# Q1: Mean, median, and std of temperatures
# Q2: What percentage of days were above 25°?
# Q3: What are the 10th and 90th percentiles?
# Q4: Show the daily changes using diff()
```

<details>
<summary>Show Answers</summary>

```python
print(f"Mean: {temps.mean():.2f}, Median: {np.median(temps):.2f}, Std: {temps.std():.2f}")

pct_above = (temps > 25).mean() * 100
print(f"Days above 25°: {pct_above:.1f}%")

print(f"10th percentile: {np.percentile(temps, 10)}")
print(f"90th percentile: {np.percentile(temps, 90)}")

print(f"Daily changes: {np.diff(temps)}")
```
</details>

---

### Exercise 4 — cumsum & unique
```python
transactions = np.array([200, -50, 300, -100, 150, -75, 400, -200, 500, -120])

# Q1: What is the running balance after each transaction?
# Q2: How many deposits (positive) and withdrawals (negative)?
# Q3: What is the final balance?

categories = np.array(['food','transport','food','entertainment',
                        'food','transport','rent','food','rent','entertainment'])
# Q4: How many times does each category appear?
```

<details>
<summary>Show Answers</summary>

```python
print(transactions.cumsum())
print(f"Deposits: {(transactions > 0).sum()}, Withdrawals: {(transactions < 0).sum()}")
print(f"Final balance: {transactions.sum()}")

cats, counts = np.unique(categories, return_counts=True)
for c, n in zip(cats, counts):
    print(f"  {c}: {n}")
```
</details>

---

## 19. Cheat Sheet

```python
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# ── BASIC AGGREGATION ─────────────────────────────────────
a.sum()                    # total sum of all elements
a.sum(axis=0)              # sum down each column
a.sum(axis=1)              # sum across each row
a.sum(axis=1, keepdims=True) # keep original dimensions

a.mean()                   # overall mean
a.mean(axis=0)             # column means
np.average(a, weights=w)   # weighted average
np.nanmean(a)              # mean ignoring NaN

# ── SPREAD ────────────────────────────────────────────────
a.std()                    # standard deviation (population)
a.std(ddof=1)              # sample standard deviation
a.var()                    # variance

# ── EXTREMES ──────────────────────────────────────────────
a.min() / a.max()          # min and max values
a.min(axis=0)              # column minimums
a.argmin() / a.argmax()    # index of min/max (flat)
a.argmin(axis=0)           # row index of min in each column
np.ptp(a)                  # range: max - min
np.nanmin(a) / np.nanmax(a) # ignore NaN

# ── PERCENTILES ───────────────────────────────────────────
np.median(a)               # 50th percentile
np.percentile(a, 75)       # 75th percentile
np.quantile(a, 0.75)       # same as above (0-1 scale)

# ── CUMULATIVE ────────────────────────────────────────────
a.cumsum()                 # running total (flattened)
a.cumsum(axis=0)           # running total down columns
a.cumprod()                # running product

# ── DIFFERENCES ───────────────────────────────────────────
np.diff(a)                 # consecutive differences (1st order)
np.diff(a, n=2)            # 2nd order differences
np.diff(a, axis=0)         # differences along rows

# ── SORTING ───────────────────────────────────────────────
np.sort(a)                 # sorted copy (ascending)
np.sort(a)[::-1]           # descending
np.sort(a, axis=0)         # sort each column
a.sort()                   # in-place sort
np.argsort(a)              # indices that would sort a

# ── UNIQUE ────────────────────────────────────────────────
np.unique(a)               # sorted unique values
np.unique(a, return_counts=True)   # values + occurrence counts
np.unique(a, return_index=True)    # values + first positions
np.unique(a, axis=0)       # unique rows

# ── DISTRIBUTION ──────────────────────────────────────────
np.histogram(a, bins=10)   # frequency counts per bin
np.digitize(a, bins)       # which bin each element falls in

# ── CORRELATION ───────────────────────────────────────────
np.corrcoef(x, y)          # Pearson correlation matrix
np.cov(x, y)               # covariance matrix

# ── CONDITIONAL COUNTING ──────────────────────────────────
(a > 5).sum()              # count elements meeting condition
(a > 5).mean()             # proportion meeting condition (0 to 1)
np.where(a > 5)            # indices where condition is True
np.unravel_index(idx, a.shape)  # flat index → (row, col)
```

---

## 🔗 What's Next?

After mastering Day 4, you're ready for:

➡️ **Day 5 — Linear Algebra & Random Module**
Learn matrix operations, solving equations, eigenvalues, and how to simulate data with NumPy's powerful random number generator.

---

## 📚 Resources

- [NumPy Statistics Docs](https://numpy.org/doc/stable/reference/routines.statistics.html)
- [NumPy Sorting & Searching](https://numpy.org/doc/stable/reference/routines.sort.html)
- [NumPy Mathematical Functions](https://numpy.org/doc/stable/reference/routines.math.html)
- [Practice on Google Colab](https://colab.research.google.com/)

---

*Part of the [7-Day NumPy Learning Plan](./README.md) · Day 4 of 7*
