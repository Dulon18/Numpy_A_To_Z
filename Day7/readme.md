# 📘 Day 7 — Capstone: Real Data Mini-Project (A to Z Guide)

A complete, end-to-end data analysis project that applies **everything from Days 1–6**. By the end of this guide, you will have built a full data pipeline from raw CSV → cleaning → analysis → insights → visualization.

---

## 📋 Table of Contents

1. [What You Will Build](#1-what-you-will-build)
2. [Tools & Setup](#2-tools--setup)
3. [Understanding the Dataset](#3-understanding-the-dataset)
4. [Step 1 — Generating the Dataset](#4-step-1--generating-the-dataset)
5. [Step 2 — Saving to CSV](#5-step-2--saving-to-csv)
6. [Step 3 — Loading the CSV](#6-step-3--loading-the-csv)
7. [Step 4 — Exploring the Data](#7-step-4--exploring-the-data)
8. [Step 5 — Cleaning the Data](#8-step-5--cleaning-the-data)
9. [Step 6 — Descriptive Statistics](#9-step-6--descriptive-statistics)
10. [Step 7 — Boolean Masks & Filtering](#10-step-7--boolean-masks--filtering)
11. [Step 8 — Grouping & Aggregation](#11-step-8--grouping--aggregation)
12. [Step 9 — Correlation Analysis](#12-step-9--correlation-analysis)
13. [Step 10 — Sorting & Ranking](#13-step-10--sorting--ranking)
14. [Step 11 — Feature Engineering](#14-step-11--feature-engineering)
15. [Step 12 — Detecting Outliers](#15-step-12--detecting-outliers)
16. [Step 13 — Visualizing with Matplotlib](#16-step-13--visualizing-with-matplotlib)
17. [Step 14 — Saving Results](#17-step-14--saving-results)
18. [Full Pipeline — One File](#18-full-pipeline--one-file)
19. [Bonus Projects](#19-bonus-projects)
20. [What to Learn Next](#20-what-to-learn-next)
21. [Complete Cheat Sheet — All 7 Days](#21-complete-cheat-sheet--all-7-days)

---

## 1. What You Will Build

A complete **Student Performance Analysis System** that:

- Generates a realistic CSV dataset of 500 students
- Loads and cleans raw data (handles missing values, bad entries)
- Computes descriptive statistics per subject and per school
- Filters students by performance thresholds
- Groups data and computes group-level summaries
- Detects outliers using IQR method
- Engineers new features (grade labels, pass/fail, GPA)
- Calculates correlations between study habits and performance
- Produces 6 publication-quality Matplotlib charts
- Saves cleaned data and summary reports

### Skills Applied

| Day | Skills Used |
|-----|-------------|
| Day 1 | Array creation, dtypes, shape |
| Day 2 | Indexing, slicing, boolean masks |
| Day 3 | Vectorized ops, broadcasting, ufuncs |
| Day 4 | Aggregation, statistics, sorting, unique |
| Day 5 | Random module for data simulation |
| Day 6 | CSV I/O, genfromtxt, savetxt, savez |
| Day 7 | Full pipeline + Matplotlib visualization |

---

## 2. Tools & Setup

### Install required libraries

```bash
pip install numpy matplotlib
```

### Import everything needed

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
```

### Check versions

```python
print(f"NumPy version:      {np.__version__}")
print(f"Matplotlib version: {plt.matplotlib.__version__}")
```

---

## 3. Understanding the Dataset

Our dataset represents **500 students** across **3 schools** with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `student_id` | int | Unique student ID (1001–1500) |
| `school_id` | int | School number (1, 2, or 3) |
| `gender` | int | 0 = Female, 1 = Male |
| `age` | int | Age in years (15–19) |
| `study_hours` | float | Weekly study hours (1–20) |
| `sleep_hours` | float | Average nightly sleep (4–10) |
| `attendance` | float | Attendance percentage (50–100) |
| `math_score` | float | Math exam score (0–100) |
| `science_score` | float | Science exam score (0–100) |
| `english_score` | float | English exam score (0–100) |
| `history_score` | float | History exam score (0–100) |

---

## 4. Step 1 — Generating the Dataset

We simulate realistic student data using NumPy's random module. Study hours, sleep, and attendance are designed to **correlate with scores** to make the analysis meaningful.

```python
import numpy as np

rng = np.random.default_rng(seed=42)
N   = 500   # number of students

# ── Basic attributes ──────────────────────────────────────────
student_ids = np.arange(1001, 1001 + N)
school_ids  = rng.integers(1, 4, N)          # schools 1, 2, 3
genders     = rng.integers(0, 2, N)          # 0=Female, 1=Male
ages        = rng.integers(15, 20, N)        # 15 to 19

# ── Study & lifestyle factors ─────────────────────────────────
study_hours  = rng.uniform(1, 20, N)         # 1–20 hrs/week
sleep_hours  = rng.uniform(4, 10, N)         # 4–10 hrs/night
attendance   = rng.uniform(50, 100, N)       # 50–100%

# ── Scores: base + contribution from study + attendance + noise ─
def make_score(base_mean, study, attend, rng, noise_std=8):
    score = (base_mean
             + study * 1.5            # study hours help
             + attend * 0.2           # attendance helps
             + rng.normal(0, noise_std, len(study)))
    return np.clip(score, 0, 100)

math_score    = make_score(40, study_hours, attendance, rng, noise_std=10)
science_score = make_score(38, study_hours, attendance, rng, noise_std=9)
english_score = make_score(45, study_hours, attendance, rng, noise_std=8)
history_score = make_score(42, study_hours, attendance, rng, noise_std=11)

# ── Introduce ~5% missing values in scores ────────────────────
def add_missing(arr, frac=0.05, rng=rng):
    arr = arr.astype(float).copy()
    idx = rng.choice(len(arr), int(len(arr) * frac), replace=False)
    arr[idx] = np.nan
    return arr

math_score    = add_missing(math_score)
science_score = add_missing(science_score)
english_score = add_missing(english_score)
history_score = add_missing(history_score)

print(f"Dataset created: {N} students")
print(f"Missing values — Math: {np.isnan(math_score).sum()}, "
      f"Science: {np.isnan(science_score).sum()}, "
      f"English: {np.isnan(english_score).sum()}, "
      f"History: {np.isnan(history_score).sum()}")
```

---

## 5. Step 2 — Saving to CSV

Stack all columns and save to disk:

```python
# Stack all columns into a 2D array (use NaN for missing scores)
data = np.column_stack([
    student_ids,
    school_ids,
    genders,
    ages,
    study_hours,
    sleep_hours,
    attendance,
    math_score,
    science_score,
    english_score,
    history_score
])

header = ('student_id,school_id,gender,age,'
          'study_hours,sleep_hours,attendance,'
          'math_score,science_score,english_score,history_score')

np.savetxt(
    'students.csv',
    data,
    delimiter=',',
    header=header,
    comments='',
    fmt=['%d', '%d', '%d', '%d',
         '%.2f', '%.2f', '%.2f',
         '%.1f', '%.1f', '%.1f', '%.1f']
)

print(f"✅ Saved students.csv  ({os.path.getsize('students.csv') / 1024:.1f} KB)")
```

---

## 6. Step 3 — Loading the CSV

Load the CSV back and split into typed columns:

```python
# Load with genfromtxt — handles NaN automatically
raw = np.genfromtxt(
    'students.csv',
    delimiter=',',
    skip_header=1,
    filling_values=np.nan
)

print(f"Loaded shape: {raw.shape}")   # (500, 11)
print(f"Total NaNs:   {np.isnan(raw).sum()}")

# ── Extract named columns ──────────────────────────────────────
student_ids   = raw[:, 0].astype(int)
school_ids    = raw[:, 1].astype(int)
genders       = raw[:, 2].astype(int)
ages          = raw[:, 3].astype(int)
study_hours   = raw[:, 4]
sleep_hours   = raw[:, 5]
attendance    = raw[:, 6]
math_score    = raw[:, 7]
science_score = raw[:, 8]
english_score = raw[:, 9]
history_score = raw[:, 10]

SUBJECTS = ['Math', 'Science', 'English', 'History']
scores   = np.column_stack([math_score, science_score,
                             english_score, history_score])

print(f"\nColumn check:")
print(f"  student_ids : {student_ids.shape}, dtype={student_ids.dtype}")
print(f"  scores      : {scores.shape},    dtype={scores.dtype}")
print(f"  study_hours : min={study_hours.min():.1f}, max={study_hours.max():.1f}")
```

---

## 7. Step 4 — Exploring the Data

Before cleaning, **always explore** your data first:

```python
print("\n" + "=" * 50)
print("  DATA EXPLORATION")
print("=" * 50)

# Shape and size
print(f"\nShape:      {raw.shape}")
print(f"Students:   {len(student_ids)}")
print(f"Features:   {raw.shape[1]}")

# School distribution
schools, school_counts = np.unique(school_ids, return_counts=True)
print("\nStudents per school:")
for s, c in zip(schools, school_counts):
    print(f"  School {s}: {c} students ({c/N*100:.1f}%)")

# Gender distribution
females = (genders == 0).sum()
males   = (genders == 1).sum()
print(f"\nGender: {females} Female ({females/N*100:.1f}%) "
      f"| {males} Male ({males/N*100:.1f}%)")

# Age distribution
ages_u, age_counts = np.unique(ages, return_counts=True)
print("\nAge distribution:")
for age, count in zip(ages_u, age_counts):
    bar = '█' * (count // 5)
    print(f"  Age {age}: {count:3d} {bar}")

# Missing values per column
col_names = ['student_id','school_id','gender','age',
             'study_hours','sleep_hours','attendance',
             'math','science','english','history']
print("\nMissing values per column:")
for name, col in zip(col_names, raw.T):
    n_missing = np.isnan(col).sum()
    if n_missing > 0:
        print(f"  {name:<15}: {n_missing} ({n_missing/N*100:.1f}%)")

# Score range overview
print("\nScore ranges (with NaN):")
for subj, col in zip(SUBJECTS, scores.T):
    valid = col[~np.isnan(col)]
    print(f"  {subj:<10}: min={valid.min():.1f}  "
          f"max={valid.max():.1f}  "
          f"mean={valid.mean():.1f}")
```

---

## 8. Step 5 — Cleaning the Data

Data cleaning is the **most important step** in any real analysis:

```python
print("\n" + "=" * 50)
print("  DATA CLEANING")
print("=" * 50)

print(f"\nBefore cleaning: {N} students")

# ── Strategy 1: Impute missing scores with subject mean ────────
# (Better than dropping — we lose less data)
for i, subj in enumerate(SUBJECTS):
    col   = scores[:, i]
    valid = col[~np.isnan(col)]
    mean  = valid.mean()
    missing_mask = np.isnan(col)
    scores[missing_mask, i] = mean
    n_imputed = missing_mask.sum()
    if n_imputed > 0:
        print(f"  Imputed {n_imputed} missing {subj} scores with mean={mean:.1f}")

# ── Strategy 2: Remove students with invalid lifestyle data ────
valid_study  = (study_hours > 0) & (study_hours <= 20)
valid_sleep  = (sleep_hours >= 3) & (sleep_hours <= 12)
valid_attend = (attendance >= 0) & (attendance <= 100)
clean_mask   = valid_study & valid_sleep & valid_attend

print(f"  Removed {(~clean_mask).sum()} rows with invalid lifestyle data")

# Apply mask
student_ids   = student_ids[clean_mask]
school_ids    = school_ids[clean_mask]
genders       = genders[clean_mask]
ages          = ages[clean_mask]
study_hours   = study_hours[clean_mask]
sleep_hours   = sleep_hours[clean_mask]
attendance    = attendance[clean_mask]
scores        = scores[clean_mask]

N_clean = len(student_ids)
print(f"\nAfter cleaning:  {N_clean} students")
print(f"Retained:        {N_clean/N*100:.1f}% of original data")

# Update subject score arrays
math_score    = scores[:, 0]
science_score = scores[:, 1]
english_score = scores[:, 2]
history_score = scores[:, 3]

# Verify no NaNs remain
assert not np.isnan(scores).any(), "Still have NaNs after cleaning!"
print("✅ No missing values remain")
```

---

## 9. Step 6 — Descriptive Statistics

Compute a full statistical summary:

```python
print("\n" + "=" * 50)
print("  DESCRIPTIVE STATISTICS")
print("=" * 50)

# ── Score statistics per subject ───────────────────────────────
print(f"\n{'Subject':<10} {'Mean':>7} {'Std':>7} {'Min':>7} "
      f"{'Q1':>7} {'Median':>8} {'Q3':>7} {'Max':>7}")
print("-" * 60)

for subj, col in zip(SUBJECTS, scores.T):
    print(f"{subj:<10} "
          f"{col.mean():>7.2f} "
          f"{col.std():>7.2f} "
          f"{col.min():>7.2f} "
          f"{np.percentile(col, 25):>7.2f} "
          f"{np.median(col):>8.2f} "
          f"{np.percentile(col, 75):>7.2f} "
          f"{col.max():>7.2f}")

# ── Overall GPA (mean of all subjects) ────────────────────────
gpa = scores.mean(axis=1)
print(f"\nOverall GPA (0–100):")
print(f"  Mean:   {gpa.mean():.2f}")
print(f"  Std:    {gpa.std():.2f}")
print(f"  Median: {np.median(gpa):.2f}")
print(f"  Min:    {gpa.min():.2f}")
print(f"  Max:    {gpa.max():.2f}")

# ── Lifestyle statistics ───────────────────────────────────────
print(f"\nLifestyle Statistics:")
print(f"  Study hours/week : {study_hours.mean():.1f} ± {study_hours.std():.1f}")
print(f"  Sleep hours/night: {sleep_hours.mean():.1f} ± {sleep_hours.std():.1f}")
print(f"  Attendance %     : {attendance.mean():.1f} ± {attendance.std():.1f}")
```

---

## 10. Step 7 — Boolean Masks & Filtering

Use boolean masks to answer targeted questions:

```python
print("\n" + "=" * 50)
print("  FILTERING & INSIGHTS")
print("=" * 50)

# ── Pass/Fail (pass = GPA >= 60) ───────────────────────────────
pass_mask = gpa >= 60
pass_count = pass_mask.sum()
fail_count = (~pass_mask).sum()
print(f"\nPass/Fail (GPA ≥ 60):")
print(f"  Pass: {pass_count} ({pass_count/N_clean*100:.1f}%)")
print(f"  Fail: {fail_count} ({fail_count/N_clean*100:.1f}%)")

# ── High achievers (GPA >= 85) ────────────────────────────────
high_mask  = gpa >= 85
print(f"\nHigh achievers (GPA ≥ 85): {high_mask.sum()} students")
print(f"  Avg study hours: {study_hours[high_mask].mean():.1f} hrs/week")
print(f"  Avg attendance:  {attendance[high_mask].mean():.1f}%")

# ── Struggling students (GPA < 50) ────────────────────────────
low_mask = gpa < 50
print(f"\nStruggling students (GPA < 50): {low_mask.sum()} students")
print(f"  Avg study hours: {study_hours[low_mask].mean():.1f} hrs/week")
print(f"  Avg attendance:  {attendance[low_mask].mean():.1f}%")

# ── Students with high study but low score (inefficient studiers)
high_study = study_hours > np.percentile(study_hours, 75)
low_gpa    = gpa < 60
inefficient = high_study & low_gpa
print(f"\nHigh study hours but low GPA: {inefficient.sum()} students")
print(f"  (May need better study strategies)")

# ── Gender performance gap ────────────────────────────────────
female_gpa = gpa[genders == 0]
male_gpa   = gpa[genders == 1]
print(f"\nGender Performance:")
print(f"  Female avg GPA: {female_gpa.mean():.2f} (n={len(female_gpa)})")
print(f"  Male avg GPA:   {male_gpa.mean():.2f} (n={len(male_gpa)})")
print(f"  Gap: {abs(female_gpa.mean() - male_gpa.mean()):.2f} points")
```

---

## 11. Step 8 — Grouping & Aggregation

Analyze performance by school and create group summaries:

```python
print("\n" + "=" * 50)
print("  GROUP ANALYSIS BY SCHOOL")
print("=" * 50)

school_list = np.unique(school_ids)

print(f"\n{'School':<10} {'N':>5} {'GPA':>8} {'Study':>8} "
      f"{'Sleep':>8} {'Attend':>8} {'Pass%':>8}")
print("-" * 58)

school_stats = []
for s in school_list:
    mask = school_ids == s
    s_gpa    = gpa[mask]
    s_study  = study_hours[mask]
    s_sleep  = sleep_hours[mask]
    s_attend = attendance[mask]
    s_pass   = (s_gpa >= 60).mean() * 100

    school_stats.append({
        'id':       s,
        'n':        mask.sum(),
        'avg_gpa':  s_gpa.mean(),
        'avg_study': s_study.mean(),
        'avg_sleep': s_sleep.mean(),
        'avg_attend': s_attend.mean(),
        'pass_rate': s_pass
    })

    print(f"School {s}   "
          f"{mask.sum():>5} "
          f"{s_gpa.mean():>8.2f} "
          f"{s_study.mean():>8.1f} "
          f"{s_sleep.mean():>8.1f} "
          f"{s_attend.mean():>8.1f} "
          f"{s_pass:>7.1f}%")

# Best performing school
best_school_idx = np.argmax([s['avg_gpa'] for s in school_stats])
print(f"\n🏆 Best performing school: "
      f"School {school_stats[best_school_idx]['id']} "
      f"(avg GPA = {school_stats[best_school_idx]['avg_gpa']:.2f})")
```

---

## 12. Step 9 — Correlation Analysis

Find which factors most strongly predict academic performance:

```python
print("\n" + "=" * 50)
print("  CORRELATION ANALYSIS")
print("=" * 50)

# Correlations with GPA
factors       = [study_hours, sleep_hours, attendance,
                 ages.astype(float), genders.astype(float)]
factor_names  = ['Study hours', 'Sleep hours', 'Attendance',
                 'Age', 'Gender']

print("\nCorrelation with GPA:")
print("-" * 35)
correlations = []
for name, factor in zip(factor_names, factors):
    corr = np.corrcoef(factor, gpa)[0, 1]
    correlations.append(corr)
    bar_len = int(abs(corr) * 30)
    direction = '+' if corr > 0 else '-'
    bar = direction * bar_len
    print(f"  {name:<15}: {corr:+.4f}  |{bar}")

strongest_idx = np.argmax(np.abs(correlations))
print(f"\n📈 Strongest predictor: {factor_names[strongest_idx]} "
      f"(r = {correlations[strongest_idx]:+.4f})")

# Subject intercorrelation matrix
print("\nSubject Score Correlations:")
corr_matrix = np.corrcoef(scores.T)
print(f"  {'':10}", end='')
for s in SUBJECTS:
    print(f"  {s[:4]:>6}", end='')
print()
for i, s1 in enumerate(SUBJECTS):
    print(f"  {s1:<10}", end='')
    for j in range(len(SUBJECTS)):
        print(f"  {corr_matrix[i,j]:>6.3f}", end='')
    print()
```

---

## 13. Step 10 — Sorting & Ranking

Rank students and identify top/bottom performers:

```python
print("\n" + "=" * 50)
print("  RANKING")
print("=" * 50)

# Rank all students by GPA (descending)
rank_idx = np.argsort(gpa)[::-1]

print("\nTop 10 Students:")
print(f"  {'Rank':<6} {'ID':>6} {'School':>8} {'GPA':>8} "
      f"{'Study hrs':>10} {'Attend%':>9}")
print("  " + "-" * 50)
for rank, idx in enumerate(rank_idx[:10], 1):
    print(f"  {rank:<6} {student_ids[idx]:>6} "
          f"{school_ids[idx]:>8} "
          f"{gpa[idx]:>8.2f} "
          f"{study_hours[idx]:>10.1f} "
          f"{attendance[idx]:>9.1f}")

print("\nBottom 5 Students:")
print(f"  {'Rank':<6} {'ID':>6} {'School':>8} {'GPA':>8} "
      f"{'Study hrs':>10} {'Attend%':>9}")
print("  " + "-" * 50)
for rank, idx in enumerate(rank_idx[-5:][::-1], 1):
    print(f"  {N_clean-5+rank:<6} {student_ids[idx]:>6} "
          f"{school_ids[idx]:>8} "
          f"{gpa[idx]:>8.2f} "
          f"{study_hours[idx]:>10.1f} "
          f"{attendance[idx]:>9.1f}")

# Percentile rank for each student
percentile_ranks = np.array([
    np.sum(gpa <= g) / N_clean * 100
    for g in gpa
])
print(f"\nPercentile ranges:")
for label, lo, hi in [('Top 10%', 90, 100), ('Middle 50%', 25, 75), ('Bottom 10%', 0, 10)]:
    mask = (percentile_ranks >= lo) & (percentile_ranks <= hi)
    print(f"  {label}: {mask.sum()} students, "
          f"avg GPA = {gpa[mask].mean():.2f}")
```

---

## 14. Step 11 — Feature Engineering

Create new meaningful features from existing data:

```python
print("\n" + "=" * 50)
print("  FEATURE ENGINEERING")
print("=" * 50)

# ── 1. Letter grades ───────────────────────────────────────────
def assign_grade(scores_arr):
    grades = np.empty(len(scores_arr), dtype='U2')
    grades[scores_arr >= 90] = 'A+'
    grades[(scores_arr >= 80) & (scores_arr < 90)] = 'A'
    grades[(scores_arr >= 70) & (scores_arr < 80)] = 'B'
    grades[(scores_arr >= 60) & (scores_arr < 70)] = 'C'
    grades[(scores_arr >= 50) & (scores_arr < 60)] = 'D'
    grades[scores_arr < 50]  = 'F'
    return grades

letter_grades = assign_grade(gpa)
grade_vals, grade_counts = np.unique(letter_grades, return_counts=True)
print("\nGrade Distribution:")
for grade, count in sorted(zip(grade_vals, grade_counts)):
    bar = '█' * (count // 5)
    print(f"  {grade:<4}: {count:>4} students  {bar}")

# ── 2. Study efficiency score (GPA per study hour) ────────────
efficiency = gpa / study_hours
print(f"\nStudy Efficiency (GPA per study hour):")
print(f"  Mean:   {efficiency.mean():.2f}")
print(f"  Best:   ID {student_ids[efficiency.argmax()]} "
      f"({efficiency.max():.2f})")
print(f"  Worst:  ID {student_ids[efficiency.argmin()]} "
      f"({efficiency.min():.2f})")

# ── 3. Risk score (combination of low study + low attend) ─────
# Normalize both to 0-1, then combine
study_norm  = (study_hours - study_hours.min()) / (study_hours.max() - study_hours.min())
attend_norm = (attendance - attendance.min()) / (attendance.max() - attendance.min())
risk_score  = 1 - (0.6 * study_norm + 0.4 * attend_norm)   # high = at risk

high_risk = risk_score > 0.7
print(f"\nAt-risk students (risk score > 0.7): {high_risk.sum()}")
print(f"  Avg GPA of at-risk students: {gpa[high_risk].mean():.2f}")

# ── 4. Subject strength (which subject each student is best at)
best_subject_idx = scores.argmax(axis=1)
print("\nStudents' strongest subject:")
for i, subj in enumerate(SUBJECTS):
    count = (best_subject_idx == i).sum()
    print(f"  {subj:<10}: {count:>4} students best in this subject")

# ── 5. Score improvement potential ────────────────────────────
# Students in bottom 25% of study hours — how much could they improve?
low_study_mask  = study_hours < np.percentile(study_hours, 25)
potential_gain  = gpa[~low_study_mask].mean() - gpa[low_study_mask].mean()
print(f"\nPotential improvement for low-study students:")
print(f"  Current avg GPA:  {gpa[low_study_mask].mean():.2f}")
print(f"  High-study avg:   {gpa[~low_study_mask].mean():.2f}")
print(f"  Potential gain:   +{potential_gain:.2f} points")
```

---

## 15. Step 12 — Detecting Outliers

Use the IQR method to find unusual scores:

```python
print("\n" + "=" * 50)
print("  OUTLIER DETECTION")
print("=" * 50)

def find_outliers(arr, name):
    Q1  = np.percentile(arr, 25)
    Q3  = np.percentile(arr, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_mask = (arr < lower) | (arr > upper)
    return outlier_mask, lower, upper

print("\nOutlier Report:")
print(f"  {'Variable':<20} {'Lower':>8} {'Upper':>8} {'Outliers':>10}")
print("  " + "-" * 50)

all_outlier_flags = []
variables   = [math_score, science_score, english_score, history_score,
               study_hours, attendance, gpa]
var_names   = ['Math score', 'Science score', 'English score',
               'History score', 'Study hours', 'Attendance', 'GPA']

for var, name in zip(variables, var_names):
    mask, lo, hi = find_outliers(var, name)
    all_outlier_flags.append(mask)
    print(f"  {name:<20} {lo:>8.2f} {hi:>8.2f} {mask.sum():>10}")

# Students who are outliers in multiple variables
outlier_count = np.sum(all_outlier_flags, axis=0)
multi_outlier = outlier_count >= 2
print(f"\nStudents who are outliers in 2+ variables: {multi_outlier.sum()}")
if multi_outlier.sum() > 0:
    print("  These students may need special attention:")
    for idx in np.where(multi_outlier)[0][:5]:
        print(f"    ID {student_ids[idx]}: GPA={gpa[idx]:.1f}, "
              f"Study={study_hours[idx]:.1f}h, "
              f"Attend={attendance[idx]:.1f}%")
```

---

## 16. Step 13 — Visualizing with Matplotlib

Create 6 informative charts:

```python
print("\n" + "=" * 50)
print("  GENERATING VISUALIZATIONS")
print("=" * 50)

fig = plt.figure(figsize=(18, 14))
fig.suptitle('Student Performance Analysis — 500 Students',
             fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

# ── Chart 1: Score distribution per subject (box plot) ────────
ax1 = fig.add_subplot(gs[0, 0])
bp  = ax1.boxplot(
    [scores[:, i] for i in range(4)],
    labels=SUBJECTS,
    patch_artist=True,
    medianprops=dict(color='black', linewidth=2)
)
for patch, color in zip(bp['boxes'], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.set_title('Score Distribution by Subject', fontweight='bold')
ax1.set_ylabel('Score')
ax1.set_ylim(0, 105)
ax1.grid(axis='y', alpha=0.3)

# ── Chart 2: GPA histogram ────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(gpa, bins=25, color='#4C72B0', alpha=0.8, edgecolor='white')
ax2.axvline(gpa.mean(), color='red', linestyle='--',
            linewidth=2, label=f'Mean = {gpa.mean():.1f}')
ax2.axvline(np.median(gpa), color='orange', linestyle='--',
            linewidth=2, label=f'Median = {np.median(gpa):.1f}')
ax2.set_title('GPA Distribution', fontweight='bold')
ax2.set_xlabel('GPA')
ax2.set_ylabel('Number of Students')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# ── Chart 3: Study hours vs GPA scatter ───────────────────────
ax3 = fig.add_subplot(gs[0, 2])
school_colors_map = {1: '#4C72B0', 2: '#DD8452', 3: '#55A868'}
for s in [1, 2, 3]:
    mask = school_ids == s
    ax3.scatter(study_hours[mask], gpa[mask],
                c=school_colors_map[s], alpha=0.5, s=15,
                label=f'School {s}')

# Trend line using least squares
A    = np.column_stack([study_hours, np.ones(N_clean)])
m, c = np.linalg.lstsq(A, gpa, rcond=None)[0]
x_line = np.linspace(study_hours.min(), study_hours.max(), 100)
ax3.plot(x_line, m * x_line + c, 'r-', linewidth=2,
         label=f'Trend (r={correlations[0]:+.3f})')
ax3.set_title('Study Hours vs GPA', fontweight='bold')
ax3.set_xlabel('Study Hours per Week')
ax3.set_ylabel('GPA')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# ── Chart 4: School comparison bar chart ──────────────────────
ax4 = fig.add_subplot(gs[1, 0])
school_gpas = [gpa[school_ids == s].mean() for s in [1, 2, 3]]
school_stds = [gpa[school_ids == s].std()  for s in [1, 2, 3]]
bars = ax4.bar(['School 1', 'School 2', 'School 3'],
               school_gpas,
               yerr=school_stds,
               color=['#4C72B0', '#DD8452', '#55A868'],
               alpha=0.8, capsize=5, edgecolor='white')
ax4.set_title('Average GPA by School (± std)', fontweight='bold')
ax4.set_ylabel('Average GPA')
ax4.set_ylim(0, 100)
ax4.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, school_gpas):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}', ha='center', va='bottom', fontsize=10,
             fontweight='bold')

# ── Chart 5: Grade distribution pie chart ─────────────────────
ax5 = fig.add_subplot(gs[1, 1])
grade_order  = ['A+', 'A', 'B', 'C', 'D', 'F']
grade_colors = ['#2ecc71', '#27ae60', '#3498db', '#f39c12', '#e67e22', '#e74c3c']
grade_cnts   = [np.sum(letter_grades == g) for g in grade_order]
non_zero     = [(g, c, col) for g, c, col in
                zip(grade_order, grade_cnts, grade_colors) if c > 0]
labels_nz, cnts_nz, cols_nz = zip(*non_zero)
wedges, texts, autotexts = ax5.pie(
    cnts_nz, labels=labels_nz, colors=cols_nz,
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=1.5)
)
for autotext in autotexts:
    autotext.set_fontsize(8)
ax5.set_title('Grade Distribution', fontweight='bold')

# ── Chart 6: Correlation heatmap ──────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
all_vars    = np.column_stack([study_hours, sleep_hours, attendance, gpa])
var_labels  = ['Study\nhours', 'Sleep\nhours', 'Attendance', 'GPA']
corr_full   = np.corrcoef(all_vars.T)

im = ax6.imshow(corr_full, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
ax6.set_xticks(range(len(var_labels)))
ax6.set_yticks(range(len(var_labels)))
ax6.set_xticklabels(var_labels, fontsize=9)
ax6.set_yticklabels(var_labels, fontsize=9)
for i in range(len(var_labels)):
    for j in range(len(var_labels)):
        ax6.text(j, i, f'{corr_full[i,j]:.2f}',
                 ha='center', va='center',
                 fontsize=9, fontweight='bold',
                 color='black')
plt.colorbar(im, ax=ax6, shrink=0.8)
ax6.set_title('Correlation Heatmap', fontweight='bold')

plt.savefig('student_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.show()
print("✅ Saved student_analysis.png")
```

---

## 17. Step 14 — Saving Results

Save cleaned data and summary report:

```python
print("\n" + "=" * 50)
print("  SAVING RESULTS")
print("=" * 50)

# ── Save cleaned data as .npz ─────────────────────────────────
np.savez_compressed(
    'students_cleaned.npz',
    student_ids   = student_ids,
    school_ids    = school_ids,
    genders       = genders,
    ages          = ages,
    study_hours   = study_hours,
    sleep_hours   = sleep_hours,
    attendance    = attendance,
    scores        = scores,
    gpa           = gpa,
    letter_grades = letter_grades,
    efficiency    = efficiency,
    risk_score    = risk_score
)
print("✅ Saved students_cleaned.npz")

# ── Save school summary as structured array ────────────────────
summary_dt = np.dtype([
    ('school_id',    'i4'),
    ('n_students',   'i4'),
    ('avg_gpa',      'f8'),
    ('avg_study',    'f8'),
    ('avg_attend',   'f8'),
    ('pass_rate',    'f8')
])

summary = np.zeros(len(school_list), dtype=summary_dt)
for i, s in enumerate(school_list):
    mask = school_ids == s
    summary[i]['school_id']  = s
    summary[i]['n_students'] = mask.sum()
    summary[i]['avg_gpa']    = gpa[mask].mean()
    summary[i]['avg_study']  = study_hours[mask].mean()
    summary[i]['avg_attend'] = attendance[mask].mean()
    summary[i]['pass_rate']  = (gpa[mask] >= 60).mean() * 100

np.save('school_summary.npy', summary)
print("✅ Saved school_summary.npy")

# ── Save top students list as CSV ─────────────────────────────
top_50_idx = rank_idx[:50]
top_data   = np.column_stack([
    student_ids[top_50_idx],
    school_ids[top_50_idx],
    np.round(gpa[top_50_idx], 2),
    np.round(study_hours[top_50_idx], 1),
    np.round(attendance[top_50_idx], 1)
])
np.savetxt(
    'top_50_students.csv',
    top_data,
    delimiter=',',
    header='student_id,school_id,gpa,study_hours,attendance',
    comments='',
    fmt=['%d', '%d', '%.2f', '%.1f', '%.1f']
)
print("✅ Saved top_50_students.csv")

# ── Print file sizes ──────────────────────────────────────────
for fname in ['students.csv', 'students_cleaned.npz',
              'school_summary.npy', 'top_50_students.csv',
              'student_analysis.png']:
    if os.path.exists(fname):
        size = os.path.getsize(fname)
        print(f"   {fname:<30}: {size/1024:.1f} KB")
```

---

## 18. Full Pipeline — One File

Here is the **complete project in one file** you can copy and run directly:

```python
"""
Day 7 Capstone — Student Performance Analysis
Run this file in Jupyter Notebook or Google Colab.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── CONFIG ───────────────────────────────────────────────────────
N    = 500
SEED = 42
SUBJECTS = ['Math', 'Science', 'English', 'History']
rng  = np.random.default_rng(SEED)

# ── 1. GENERATE DATA ─────────────────────────────────────────────
student_ids = np.arange(1001, 1001 + N)
school_ids  = rng.integers(1, 4, N)
genders     = rng.integers(0, 2, N)
ages        = rng.integers(15, 20, N)
study_hours = rng.uniform(1, 20, N)
sleep_hours = rng.uniform(4, 10, N)
attendance  = rng.uniform(50, 100, N)

def make_score(base, study, attend, rng, noise=8):
    s = base + study * 1.5 + attend * 0.2 + rng.normal(0, noise, len(study))
    return np.clip(s, 0, 100)

raw_scores = np.column_stack([
    make_score(40, study_hours, attendance, rng, 10),
    make_score(38, study_hours, attendance, rng, 9),
    make_score(45, study_hours, attendance, rng, 8),
    make_score(42, study_hours, attendance, rng, 11)
]).astype(float)

for i in range(4):
    idx = rng.choice(N, int(N * 0.05), replace=False)
    raw_scores[idx, i] = np.nan

# ── 2. SAVE CSV ───────────────────────────────────────────────────
data = np.column_stack([student_ids, school_ids, genders, ages,
                        study_hours, sleep_hours, attendance, raw_scores])
np.savetxt('students.csv', data, delimiter=',',
    header='student_id,school_id,gender,age,study_hours,sleep_hours,attendance,math,science,english,history',
    comments='',
    fmt=['%d','%d','%d','%d','%.2f','%.2f','%.2f','%.1f','%.1f','%.1f','%.1f'])

# ── 3. LOAD ───────────────────────────────────────────────────────
raw = np.genfromtxt('students.csv', delimiter=',', skip_header=1, filling_values=np.nan)
student_ids   = raw[:,0].astype(int)
school_ids    = raw[:,1].astype(int)
genders       = raw[:,2].astype(int)
ages          = raw[:,3].astype(int)
study_hours   = raw[:,4]
sleep_hours   = raw[:,5]
attendance    = raw[:,6]
scores        = raw[:,7:]

# ── 4. CLEAN ─────────────────────────────────────────────────────
for i in range(4):
    col = scores[:, i]
    col[np.isnan(col)] = col[~np.isnan(col)].mean()

valid = (study_hours > 0) & (study_hours <= 20) & \
        (sleep_hours >= 3) & (sleep_hours <= 12) & \
        (attendance >= 0) & (attendance <= 100)

student_ids = student_ids[valid]; school_ids = school_ids[valid]
genders     = genders[valid];     ages       = ages[valid]
study_hours = study_hours[valid]; sleep_hours = sleep_hours[valid]
attendance  = attendance[valid];  scores      = scores[valid]
N_clean     = len(student_ids)

# ── 5. COMPUTE FEATURES ───────────────────────────────────────────
gpa        = scores.mean(axis=1)
efficiency = gpa / study_hours

def assign_grade(arr):
    g = np.empty(len(arr), dtype='U2')
    g[arr >= 90] = 'A+'; g[(arr>=80)&(arr<90)] = 'A'
    g[(arr>=70)&(arr<80)] = 'B'; g[(arr>=60)&(arr<70)] = 'C'
    g[(arr>=50)&(arr<60)] = 'D'; g[arr < 50] = 'F'
    return g

letter_grades = assign_grade(gpa)

# ── 6. STATISTICS ────────────────────────────────────────────────
print("\n── Score Statistics ──")
for subj, col in zip(SUBJECTS, scores.T):
    print(f"{subj:<10}: mean={col.mean():.1f} std={col.std():.1f} "
          f"min={col.min():.1f} max={col.max():.1f}")

corr_study  = np.corrcoef(study_hours, gpa)[0,1]
corr_attend = np.corrcoef(attendance,  gpa)[0,1]
print(f"\nCorrelations with GPA:")
print(f"  Study hours: {corr_study:+.4f}")
print(f"  Attendance:  {corr_attend:+.4f}")

print(f"\nPass rate (GPA≥60): {(gpa>=60).mean()*100:.1f}%")

# ── 7. VISUALIZE ─────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Student Performance Analysis', fontsize=16, fontweight='bold')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
COLORS = ['#4C72B0','#DD8452','#55A868','#C44E52']

ax1 = fig.add_subplot(gs[0,0])
bp  = ax1.boxplot([scores[:,i] for i in range(4)], labels=SUBJECTS,
                  patch_artist=True, medianprops=dict(color='black', linewidth=2))
for patch, c in zip(bp['boxes'], COLORS):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax1.set_title('Score Distribution'); ax1.set_ylabel('Score'); ax1.grid(axis='y', alpha=0.3)

ax2 = fig.add_subplot(gs[0,1])
ax2.hist(gpa, bins=25, color='#4C72B0', alpha=0.8, edgecolor='white')
ax2.axvline(gpa.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={gpa.mean():.1f}')
ax2.axvline(np.median(gpa), color='orange', linestyle='--', linewidth=2, label=f'Median={np.median(gpa):.1f}')
ax2.set_title('GPA Distribution'); ax2.set_xlabel('GPA'); ax2.legend(); ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(gs[0,2])
for s, c in zip([1,2,3], COLORS):
    m = school_ids == s
    ax3.scatter(study_hours[m], gpa[m], c=c, alpha=0.5, s=15, label=f'School {s}')
A_ls = np.column_stack([study_hours, np.ones(N_clean)])
m_ls, c_ls = np.linalg.lstsq(A_ls, gpa, rcond=None)[0]
x_l = np.linspace(study_hours.min(), study_hours.max(), 100)
ax3.plot(x_l, m_ls*x_l+c_ls, 'r-', lw=2, label=f'r={corr_study:+.3f}')
ax3.set_title('Study Hours vs GPA'); ax3.set_xlabel('Study hrs/week')
ax3.set_ylabel('GPA'); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

ax4 = fig.add_subplot(gs[1,0])
sg  = [gpa[school_ids==s].mean() for s in [1,2,3]]
ss  = [gpa[school_ids==s].std()  for s in [1,2,3]]
bars = ax4.bar(['School 1','School 2','School 3'], sg, yerr=ss,
               color=COLORS[:3], alpha=0.8, capsize=5, edgecolor='white')
ax4.set_title('Avg GPA by School (±std)'); ax4.set_ylabel('GPA')
ax4.set_ylim(0,100); ax4.grid(axis='y', alpha=0.3)
for b, v in zip(bars, sg):
    ax4.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{v:.1f}',
             ha='center', fontsize=10, fontweight='bold')

ax5 = fig.add_subplot(gs[1,1])
g_order  = ['A+','A','B','C','D','F']
g_colors = ['#2ecc71','#27ae60','#3498db','#f39c12','#e67e22','#e74c3c']
g_cnts   = [np.sum(letter_grades==g) for g in g_order]
nz = [(g,c,col) for g,c,col in zip(g_order,g_cnts,g_colors) if c>0]
lg, lc, lco = zip(*nz)
ax5.pie(lc, labels=lg, colors=lco, autopct='%1.1f%%',
        wedgeprops=dict(edgecolor='white'))
ax5.set_title('Grade Distribution')

ax6 = fig.add_subplot(gs[1,2])
av = np.column_stack([study_hours, sleep_hours, attendance, gpa])
vl = ['Study\nhrs','Sleep\nhrs','Attend','GPA']
cm = np.corrcoef(av.T)
im = ax6.imshow(cm, cmap='RdYlGn', vmin=-1, vmax=1)
ax6.set_xticks(range(4)); ax6.set_yticks(range(4))
ax6.set_xticklabels(vl, fontsize=9); ax6.set_yticklabels(vl, fontsize=9)
for i in range(4):
    for j in range(4):
        ax6.text(j, i, f'{cm[i,j]:.2f}', ha='center', va='center',
                 fontsize=9, fontweight='bold')
plt.colorbar(im, ax=ax6, shrink=0.8)
ax6.set_title('Correlation Heatmap')

plt.savefig('student_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

# ── 8. SAVE RESULTS ──────────────────────────────────────────────
np.savez_compressed('students_cleaned.npz',
    student_ids=student_ids, school_ids=school_ids,
    genders=genders, ages=ages, study_hours=study_hours,
    sleep_hours=sleep_hours, attendance=attendance,
    scores=scores, gpa=gpa, letter_grades=letter_grades)

print("\n✅ Complete pipeline finished!")
print("   Files: students.csv, students_cleaned.npz, student_analysis.png")
```

---

## 19. Bonus Projects

Now that you have mastered the core pipeline, here are **3 bonus projects** to push further:

### Bonus 1 — Grade Prediction

```python
# Predict grade (pass/fail) from study habits using a simple threshold model
# Try different thresholds and find the one that maximizes accuracy

thresholds = np.arange(5, 20, 0.5)
accuracies = []

for t in thresholds:
    pred_pass   = study_hours >= t          # predict pass if study >= t
    actual_pass = gpa >= 60
    accuracy    = (pred_pass == actual_pass).mean()
    accuracies.append(accuracy)

best_threshold = thresholds[np.argmax(accuracies)]
print(f"Best study-hour threshold: {best_threshold:.1f} hrs/week")
print(f"Prediction accuracy:       {max(accuracies)*100:.1f}%")
```

### Bonus 2 — Bootstrap Confidence Intervals

```python
# Estimate 95% confidence interval for mean GPA using bootstrap resampling
rng = np.random.default_rng(42)
n_bootstrap = 10000
boot_means  = np.array([
    rng.choice(gpa, size=len(gpa), replace=True).mean()
    for _ in range(n_bootstrap)
])

ci_low  = np.percentile(boot_means, 2.5)
ci_high = np.percentile(boot_means, 97.5)
print(f"Mean GPA: {gpa.mean():.2f}")
print(f"95% CI:   [{ci_low:.2f}, {ci_high:.2f}]")
```

### Bonus 3 — Monte Carlo Enrollment Simulation

```python
# Simulate 1000 different possible cohorts of 500 students
# and find the distribution of pass rates

rng = np.random.default_rng(42)
n_simulations = 1000
simulated_pass_rates = np.zeros(n_simulations)

for sim in range(n_simulations):
    sh    = rng.uniform(1, 20, 500)
    att   = rng.uniform(50, 100, 500)
    score = 40 + sh * 1.5 + att * 0.2 + rng.normal(0, 10, 500)
    score = np.clip(score, 0, 100)
    simulated_pass_rates[sim] = (score >= 60).mean() * 100

print(f"Simulated pass rate: {simulated_pass_rates.mean():.1f}% "
      f"± {simulated_pass_rates.std():.1f}%")
print(f"95% range: [{np.percentile(simulated_pass_rates, 2.5):.1f}%, "
      f"{np.percentile(simulated_pass_rates, 97.5):.1f}%]")
```

---

## 20. What to Learn Next

Congratulations on completing the 7-Day NumPy plan! 🎉

Here is your recommended learning roadmap:

```
NumPy (Done ✅)
    │
    ├── Pandas          → Working with labeled DataFrames, time series, groupby
    │   pip install pandas
    │
    ├── Matplotlib      → Advanced charting, subplots, customization
    │   pip install matplotlib
    │
    ├── SciPy           → Scientific computing: optimization, signal processing, stats
    │   pip install scipy
    │
    ├── Scikit-learn    → Machine learning: regression, classification, clustering
    │   pip install scikit-learn
    │
    └── TensorFlow /    → Deep learning: neural networks
        PyTorch         pip install tensorflow  OR  pip install torch
```

### NumPy topics to explore next

- `np.fft` — Fast Fourier Transform for signal processing
- `np.polynomial` — polynomial fitting and evaluation
- `np.ma` — masked arrays (alternative to NaN handling)
- Cython / Numba — compiling NumPy-style code for even faster execution
- Memory-mapped arrays (`np.memmap`) — working with files larger than RAM

---

## 21. Complete Cheat Sheet — All 7 Days

```python
import numpy as np

# ══ DAY 1 — ARRAY CREATION ═══════════════════════════════════
np.array([1,2,3])                  # from list
np.zeros((m, n))                   # zeros
np.ones((m, n))                    # ones
np.eye(n)                          # identity
np.arange(start, stop, step)       # range
np.linspace(start, stop, num)      # evenly spaced
np.full((m,n), val)                # filled with val
a.shape; a.ndim; a.size; a.dtype   # properties

# ══ DAY 2 — INDEXING & RESHAPING ════════════════════════════
a[i]           # 1D index
a[i, j]        # 2D index
a[-1]          # last element
a[1:4]         # slice
a[::2]         # every other
a[::-1]        # reversed
a[a > 5]       # boolean mask
a[[0,2,4]]     # fancy indexing
a.reshape(m,n) # reshape
a.flatten()    # 1D copy
a.ravel()      # 1D view
a.T            # transpose

# ══ DAY 3 — VECTORIZATION & BROADCASTING ════════════════════
a + b; a - b; a * b; a / b; a ** 2  # element-wise
a + scalar                           # broadcast scalar
a + np.array([[1],[2],[3]])          # broadcast column
np.sqrt(a); np.abs(a); np.exp(a)     # ufuncs
np.sin(a); np.cos(a); np.log(a)
np.clip(a, lo, hi)                   # clamp
np.where(cond, x, y)                 # conditional select

# ══ DAY 4 — AGGREGATION & STATISTICS ════════════════════════
a.sum(); a.sum(axis=0); a.sum(axis=1)
a.mean(); np.nanmean(a)
a.std(); a.std(ddof=1)
a.min(); a.max()
a.argmin(); a.argmax()
np.median(a); np.percentile(a, 75)
a.cumsum(); np.diff(a)
np.sort(a); np.argsort(a)
np.unique(a, return_counts=True)
np.corrcoef(x, y); np.cov(x, y)
(a > 5).sum(); (a > 5).mean()

# ══ DAY 5 — LINEAR ALGEBRA & RANDOM ════════════════════════
A @ B                                # matrix multiply
np.linalg.inv(A)                     # inverse
np.linalg.det(A)                     # determinant
np.linalg.solve(A, b)                # solve Ax=b
np.linalg.eig(A)                     # eigenvalues/vectors
np.linalg.norm(v)                    # L2 norm
np.linalg.svd(A)                     # SVD decomposition
np.linalg.matrix_rank(A)             # rank
np.linalg.lstsq(A, b, rcond=None)    # least squares

rng = np.random.default_rng(42)
rng.random(n)                        # uniform [0,1)
rng.normal(mean, std, n)             # Gaussian
rng.integers(lo, hi, n)             # random ints
rng.choice(arr, n, replace=False)    # sampling
rng.shuffle(a); rng.permutation(a)   # shuffling
rng.binomial(n, p, size)             # binomial
rng.poisson(lam, size)               # Poisson

# ══ DAY 6 — I/O & MEMORY ════════════════════════════════════
np.save('f.npy', arr)                # save binary
np.load('f.npy')                     # load binary
np.savez('f.npz', a=arr1, b=arr2)   # save multiple
np.savetxt('f.csv', arr,            # save CSV
    delimiter=',', fmt='%.4f',
    header='c1,c2', comments='')
np.loadtxt('f.csv', delimiter=',', skiprows=1)
np.genfromtxt('f.csv', delimiter=',',
    names=True, filling_values=np.nan)

dt = np.dtype([('name','U10'),('age','i4')])
arr = np.array([('Alice',25)], dtype=dt)
arr['name']; arr[arr['age']>30]      # structured access

arr.astype(np.float32)               # convert dtype (copy)
arr.nbytes; arr.itemsize             # memory info
np.shares_memory(a, b)              # check view/copy
arr.flags['C_CONTIGUOUS']           # check layout
np.ascontiguousarray(arr)           # make C-contiguous
arr += 1                             # in-place op
np.add(a, b, out=result)            # avoid temporaries

# ══ DAY 7 — FULL PIPELINE PATTERNS ══════════════════════════
# Generate → Save → Load → Clean → Analyze → Visualize → Save

# Impute missing values
col[np.isnan(col)] = col[~np.isnan(col)].mean()

# Clean with boolean mask
valid = (arr > lo) & (arr < hi)
arr   = arr[valid]

# Group aggregation
for group in np.unique(group_ids):
    mask = group_ids == group
    print(arr[mask].mean())

# Outlier detection (IQR method)
Q1, Q3 = np.percentile(arr, [25, 75])
IQR     = Q3 - Q1
outliers = (arr < Q1-1.5*IQR) | (arr > Q3+1.5*IQR)

# Feature normalization (min-max)
norm = (arr - arr.min()) / (arr.max() - arr.min())

# Least squares trend line
A    = np.column_stack([x, np.ones(len(x))])
m, c = np.linalg.lstsq(A, y, rcond=None)[0]

# Bootstrap confidence interval
boot = np.array([rng.choice(arr,len(arr),True).mean()
                 for _ in range(10000)])
ci   = np.percentile(boot, [2.5, 97.5])
```

---

## 🎉 Congratulations!

You have completed the full **7-Day NumPy Learning Plan**. Here is what you now know:

| Day | Topic | Key Skills |
|-----|-------|------------|
| ✅ Day 1 | Array basics | Create, inspect, dtypes |
| ✅ Day 2 | Indexing & reshaping | Slice, mask, reshape, transpose |
| ✅ Day 3 | Vectorization | Broadcasting, ufuncs, np.where |
| ✅ Day 4 | Aggregation | Stats, sorting, unique, correlation |
| ✅ Day 5 | Linear algebra & random | Solve equations, simulate data |
| ✅ Day 6 | I/O & memory | CSV, binary, structured arrays, dtype |
| ✅ Day 7 | Capstone project | Full pipeline, visualization |

---

## 📚 Resources

- [NumPy Official Docs](https://numpy.org/doc/)
- [Matplotlib Docs](https://matplotlib.org/stable/contents.html)
- [100 NumPy Exercises](https://github.com/rougier/numpy-100)
- [NumPy Illustrated](https://betterprogramming.pub/numpy-illustrated-the-visual-guide-to-numpy-3b1d4976de1d)
- [Practice on Google Colab](https://colab.research.google.com/)

---

*Part of the [7-Day NumPy Learning Plan](./README.md) · Day 7 of 7 — COMPLETE! 🚀*
