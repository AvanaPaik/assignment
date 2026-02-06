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

# Python Fundamentals: Introduction to Databases (SQL, NoSQL & Vector Databases)

## 1. What Is a Database?

A **database** is an organized collection of data stored electronically
and managed by a Database Management System (DBMS).

Why databases are used: - Efficient storage - Fast retrieval - Data
integrity - Security - Concurrent access

------------------------------------------------------------------------

## 2. Types of Databases

Main categories: - **Relational (SQL) Databases** - **NoSQL
Databases** - **Vector Databases**

------------------------------------------------------------------------

# 3. SQL (Relational) Databases

SQL = Structured Query Language.

Examples: - MySQL - PostgreSQL - SQLite - Oracle - SQL Server

### 3.1 Key Features

-   Data stored in **tables**
-   Rows and columns
-   Fixed schema
-   Relationships using keys
-   ACID transactions

### 3.2 Basic Terminology

-   Table
-   Row / Record
-   Column / Field
-   Primary Key
-   Foreign Key

------------------------------------------------------------------------

## 3.3 Example Table

  id   name   age
  ---- ------ -----
  1    Riya   20
  2    Aman   22

------------------------------------------------------------------------

## 3.4 Basic SQL Commands

### Create Table

``` sql
CREATE TABLE students (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER
);
```

### Insert Data

``` sql
INSERT INTO students VALUES (1, 'Riya', 20);
```

### Query Data

``` sql
SELECT * FROM students;
```

### Filter

``` sql
SELECT name FROM students WHERE age > 21;
```

### Update

``` sql
UPDATE students SET age = 23 WHERE id = 2;
```

### Delete

``` sql
DELETE FROM students WHERE id = 1;
```

------------------------------------------------------------------------

## 4. NoSQL Databases

NoSQL = Not Only SQL.

Used for: - Big data - Distributed systems - Flexible schemas -
Real-time apps

Examples: - MongoDB (Document) - Redis (Key-Value) - Cassandra (Wide
Column) - Neo4j (Graph)

------------------------------------------------------------------------

## 4.1 Types of NoSQL Databases

### Document-Based

Stores data as JSON-like documents. Example:

``` json
{
  "id": 1,
  "name": "Riya",
  "age": 20
}
```

### Key--Value Stores

    "user:101" → "active"

### Column-Family

Data grouped by columns.

### Graph Databases

Nodes + Edges + Relationships.

------------------------------------------------------------------------

## 4.2 SQL vs NoSQL

  Feature        SQL            NoSQL
  -------------- -------------- ---------------------
  Schema         Fixed          Flexible
  Scaling        Vertical       Horizontal
  Structure      Tables         Documents/Key-Value
  Transactions   Strong         Often eventual
  Use Case       Banking, ERP   Big data, web apps

------------------------------------------------------------------------

# 5. Vector Databases

Vector databases store **embeddings**---numerical representations of
text, images, audio, etc.

Used in: - AI search - Recommendation systems - Chatbots - RAG systems
(Retrieval-Augmented Generation)

Examples: - Pinecone - Weaviate - Milvus - FAISS - Chroma

------------------------------------------------------------------------

## 5.1 What Is an Embedding?

An embedding is a list of numbers representing meaning:

    [0.12, -0.45, 0.88, ...]

Similar data → closer vectors.

------------------------------------------------------------------------

## 5.2 Key Operations

-   Insert vectors
-   Similarity search
-   Nearest neighbor queries
-   Indexing

Distance metrics: - Cosine similarity - Euclidean distance - Dot product

------------------------------------------------------------------------

## 6. Using Databases in Python

### 6.1 SQL with SQLite

``` python
import sqlite3

conn = sqlite3.connect("school.db")
cur = conn.cursor()

cur.execute("SELECT * FROM students")
rows = cur.fetchall()

conn.close()
```

------------------------------------------------------------------------

### 6.2 NoSQL with MongoDB (pymongo)

``` python
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client.school

students = db.students.find()
for s in students:
    print(s)
```

------------------------------------------------------------------------

### 6.3 Vector DB Example (Conceptual)

``` python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

embedding = model.encode("Databases are useful")

# Store embedding in vector DB
# db.insert(vector=embedding)
```

------------------------------------------------------------------------

## 7. When to Use Which?

Use SQL when: - Data is structured - Relationships matter - Transactions
are critical

Use NoSQL when: - Data changes often - High scalability is required -
Distributed apps

Use Vector DB when: - Semantic search needed - AI/ML systems -
Recommendation engines

------------------------------------------------------------------------

## 8. Practice Questions

1.  What is a relational database?
2.  List two SQL databases.
3.  Difference between SQL and NoSQL.
4.  What is a vector database used for?
5.  Name two vector databases.
6.  Write a SQL query to select students older than 18.
7.  Why are embeddings important?

------------------------------------------------------------------------

## 9. Quick Revision Sheet

-   SQL → tables, schema, joins
-   NoSQL → flexible, distributed
-   Vector DB → similarity search
-   Embeddings → numeric meaning
-   Python connectors → sqlite3, pymongo

------------------------------------------------------------------------

**End of Notes**
