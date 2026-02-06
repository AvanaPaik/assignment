# assignment
# Python Fundamentals: Data Loading, Wrangling & Visualization

## 1. Introduction

In data science and analytics, Python is widely used for: - Loading
datasets from files or the web - Cleaning and transforming data
(wrangling) - Visualizing patterns and trends

Key libraries: - **pandas** -- data manipulation - **numpy** --
numerical computing - **matplotlib / seaborn** -- visualization

------------------------------------------------------------------------

## 2. Data Loading in Python

### 2.1 Loading CSV Files

``` python
import pandas as pd

df = pd.read_csv("data.csv")
print(df.head())
```

### 2.2 Loading Excel Files

``` python
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")
```

### 2.3 Loading from URLs

``` python
url = "https://example.com/data.csv"
df = pd.read_csv(url)
```

### 2.4 Other Formats

-   JSON: `pd.read_json()`
-   SQL: `pd.read_sql()`
-   TXT/TSV: `pd.read_csv("file.txt", sep="\t")`

------------------------------------------------------------------------

## 3. Inspecting Data

### Basic Information

``` python
df.shape
df.columns
df.dtypes
df.info()
df.describe()
```

### Viewing Data

``` python
df.head()
df.tail()
df.sample(5)
```

------------------------------------------------------------------------

## 4. Data Wrangling (Cleaning & Transformation)

### 4.1 Handling Missing Values

``` python
df.isnull().sum()
df.dropna()
df.fillna(0)
```

### 4.2 Renaming Columns

``` python
df.rename(columns={"old": "new"}, inplace=True)
```

### 4.3 Changing Data Types

``` python
df["age"] = df["age"].astype(int)
```

### 4.4 Filtering Rows

``` python
df[df["score"] > 80]
```

### 4.5 Sorting

``` python
df.sort_values(by="score", ascending=False)
```

### 4.6 Creating New Columns

``` python
df["total"] = df["math"] + df["science"]
```

------------------------------------------------------------------------

## 5. Grouping and Aggregation

``` python
df.groupby("category")["sales"].mean()
df.groupby("region").agg({
    "sales": "sum",
    "profit": "mean"
})
```

------------------------------------------------------------------------

## 6. Merging and Joining Data

### Concatenation

``` python
pd.concat([df1, df2])
```

### Merge / Join

``` python
pd.merge(df1, df2, on="id", how="inner")
```

Types of joins: - inner - left - right - outer

------------------------------------------------------------------------

## 7. Data Visualization Basics

### 7.1 Using Matplotlib

``` python
import matplotlib.pyplot as plt

plt.plot(df["year"], df["sales"])
plt.xlabel("Year")
plt.ylabel("Sales")
plt.title("Sales Over Time")
plt.show()
```

### 7.2 Bar Chart

``` python
plt.bar(df["product"], df["profit"])
plt.show()
```

### 7.3 Histogram

``` python
plt.hist(df["age"], bins=10)
plt.show()
```

------------------------------------------------------------------------

## 8. Visualization with Seaborn

``` python
import seaborn as sns

sns.lineplot(x="year", y="sales", data=df)
sns.boxplot(x="category", y="price", data=df)
```

------------------------------------------------------------------------

## 9. Simple End‑to‑End Example

``` python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("students.csv")

# Clean
df.fillna(0, inplace=True)

# Filter
top = df[df["marks"] > 75]

# Group
avg = df.groupby("class")["marks"].mean()

# Plot
avg.plot(kind="bar")
plt.title("Average Marks by Class")
plt.show()
```

------------------------------------------------------------------------

## 10. Practice Exercises

1.  Load a CSV file and display its first 10 rows.
2.  Find missing values in each column.
3.  Replace missing values with the column mean.
4.  Filter rows where age \> 25.
5.  Group by department and calculate average salary.
6.  Plot a histogram of salaries.
7.  Merge two datasets using a common column.

------------------------------------------------------------------------

## 11. Quick Revision Sheet

-   `read_csv()` → load data
-   `info()` → dataset summary
-   `dropna()` / `fillna()` → missing values
-   `groupby()` → aggregation
-   `merge()` → combine datasets
-   `plot()` / `sns.*()` → visualize

------------------------------------------------------------------------

**End of Notes**
