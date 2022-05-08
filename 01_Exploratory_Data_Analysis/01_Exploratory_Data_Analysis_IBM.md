# Exploratory Data Analysis

These are my notes and the code of the [IBM Machine Learning Professional Certificate](https://www.coursera.org/professional-certificates/ibm-machine-learning) offered by IBM & Coursera.

The Specialization is divided in 6 courses, and each of them has its own folder with its guide & notebooks:

1. [Exploratory Data Analysis for Machine Learning](https://www.coursera.org/learn/ibm-exploratory-data-analysis-for-machine-learning?specialization=ibm-machine-learning)
2. [Supervised Machine Learning: Regression](https://www.coursera.org/learn/supervised-machine-learning-regression?specialization=ibm-machine-learning)
3. [Supervised Machine Learning: Classification](https://www.coursera.org/learn/supervised-machine-learning-classification?specialization=ibm-machine-learning)
4. [Unsupervised Machine Learning](https://www.coursera.org/learn/ibm-unsupervised-machine-learning?specialization=ibm-machine-learning)
5. [Deep Learning and Reinforcement Learning](https://www.coursera.org/learn/deep-learning-reinforcement-learning?specialization=ibm-machine-learning)
6. [Specialized Models: Time Series and Survival Analysis](https://www.coursera.org/learn/time-series-survival-analysis?specialization=ibm-machine-learning)

This file focuses on the **first course: Exploratory Data Analysis for Machine Learning**

Mikel Sagardia, 2022.
No guarantees

## Overview of Contents

1. A Brief History of Modern AI and its Applications (Week 1)
2. Retrieving and Cleaning Data (Week 2)
3. Exploratory Data Analysis and Feature Engineering (Week 3)
4. Inferential Statistics and Hypothesis Testing (Week 4)
5. (Optional) HONORS Project (Week 5)

## 1. A Brief History of Modern AI and its Applications (Week 1)

This section is very general and almost nothing new is explained.

Deep Learning in Machine Learning in Artificial Intelligence.

Two important AI breakthroughs:

- Image classification: since 2015, computers better than humans
- Machine Translation: since 2016, near human performance

AI: Simulation of intelligence behavior; mimicking of human cognitive capabilities. AI programs can sense, reason, act, and adapt.

Machine Learning: We learn patterns as we are exposed to more data; we don't program the patterns, but we detect them from the data.

Supervised learning vs. Unsupervised learning: make predictions vs. find structure.

Deep Learning: features detected automatically, complex neural networks used.

Examples given:

- Email spam classification
- Market segmentation
- Iris flower classification
- Face classification

There have been several AI winters, because the expectations were not met. However, we are living a golden phase again, since two major important points due to Deep Learning:

- Image classification: AlexNet, 2012 (Hinton)
- Language understanding: Word2Vec, 2013 (Mikolov)

Since then, many things are happening faster:

- Tensorflow, 2015
- AlphaGo, 2016
- Waymo self-driving car, 2018
- etc.

Modern AI: Impactful Areas:

- Object detection for self-driving cars (CV)
- Healthcare: disease detection (CV)
- Language translation (NLP)

What is different now?

- Bigger datasets
- Faster computers, GPUs
- Better algorithms: Neural Nets

Many applications are mentioned.

Machine Learning Workflow:

- Problem statement
- Data collection
- Data exploration and preprocessing
- Modeling
- Validation
- Decision Making and Deployment

![Machine Learning Workflow](./pics/ml_workflow.png)

Machine Learning Vocabulary:

- Target: what we want to predict.
- Features: explanatory variables.
- Example: observation, a single row, a data-point.
- Label: target value of a data-point.

## 2. Retrieving and Cleaning Data (Week 2)

### 2.1 Retrieving Data: CSV, JSON, SQL, NoSQL, APIs

#### CSV

```python
import pandas as pd

data = pd.read_csv("data/file.csv")

# Other separators: TAB
data = pd.read_csv("data/file.csv", sep="\t")
data = pd.read_csv("data/file.csv", delim_whitespace=True)

# First row not names
data = pd.read_csv("data/file.csv", header=None)

# Set column names
data = pd.read_csv("data/file.csv", sep="\t")

# Specify colunmn names
data = pd.read_csv("data/file.csv", names=['Name1', 'Name2'])

# Custom missing names: replace 99 with NA
data = pd.read_csv("data/file.csv", na_values=['NA', 99])
```

#### JSON

```python
import pandas as pd
filepath = ""
data = pd.read_json("data/file.json")
data = pd.to_json("data/ouput.json")
```

#### SQL

There are many libraries which interact with SQL or relational databases: PostgreSQL, MySQL, SQLite, etc. Each one with its specific properties.

```python
import sqlite3 as sq3
#import pandas.io.sql as pds
import pandas as pd

# Path to database
path = 'data/classic_rock.db'
con = sq3.Connection(path)

# Write the query
query = '''
SELECT * 
FROM rock_songs;
'''

# Execute the query
#observations = pds.read_sql(query, con)
data = pd.read_sql(query, con)
```

#### NoSQL

NoSQL databases are non-relational databases: the data is not structured in tables. There are several types: document-based, key-value-pair-based, etc.

MongoDB is an example of document-based NoSQL database. Each entry is a document, which is interfaces as a JSON object.

```python
from pymongo import MongoClient
con = MongoClient()

# Get list of available databases
con.list_database_names()
# Choose database with name database_name
db = con.database_name

# Create a cursor object using a query
# Similarly to how we have multiple tables
# in Mongo we have multiple collection_names -> we need to select one
# query: MongoDB query string, or `{}` to select all
cursor = db.collection_name.find(query)

# Expand cursor and construct DataFrame
# list(cursor) is a list of python dictionaries
# which are converted into a Dataframe
df = pd.DataFrame(list(cursor))
```

#### APIs & Cloud

```python
data_url = 'https://.../database.data'
df = pw.read_csv(data_url)
```

### 2.2 Lab Notebooks: Retrieving Data

#### `./lab/01a_DEMO_Reading_Data.ipynb`

An example of how to interact with SQL databases.

```python
# Imports
import sqlite3 as sq3
import pandas.io.sql as pds
import pandas as pd

# Download the database
!wget -P data https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/classic_rock.db

# Initialize path to SQLite databasejdbc:sqlite:/C:/__tmp/test/sqlite/jdbcTest.db
# Define path to database + create connection
path = 'data/classic_rock.db'
con = sq3.Connection(path)

# Write the query
query = '''
SELECT * 
FROM rock_songs;
'''

# Execute the query
observations = pds.read_sql(query, con)
observations.head()

# More complex query
# executed with a generator using some options
query='''
SELECT Artist, Release_Year, COUNT(*) AS num_songs, AVG(PlayCount) AS avg_plays  
    FROM rock_songs
    GROUP BY Artist, Release_Year
    ORDER BY num_songs desc;
'''

# Execute the query with a generator
# and interesting options
observations_generator = pds.read_sql(query,
                            con,
                            coerce_float=True, # Doesn't efefct this dataset, because floats were correctly parsed
                            parse_dates=['Release_Year'], # Parse `Release_Year` as a date
                            chunksize=5 # Allows for streaming results as a series of shorter tables
                           )

for index, observations in enumerate(observations_generator):
    if index < 5:
        print(f'Observations index: {index}')
        display(observations)

```

#### `./lab/01b_LAB_Reading_Data.ipynb`

Similar to `./lab/01a_DEMO_Reading_Data.ipynb`.

### 2.3 Data Cleaning

Garbage in, garbage out. So we need to have high quality data.

Most important issues of data:
- Diplicates
- Inconsistent text/typos in labels
- Missing data
- Outliers that yield skewed models
- Sourcing issues: multiple technologies that need to be synched

#### Missing Data

Policies or actions we can take:
- Remove the row with a missing field/column value; however, the complete row is deleted, i.e., also its fields which are not missing. We might remove valuable information!
- Impute the missing field/column: replace with mean, meadian, most common, etc. However, we are creating new artificial values.
- Mask the data: create a category for missing values. Maybe there is valueble information in the fact that it's missing. However, we categorize all missing values as the same, and the underlying causes of them being missing might be different!

#### Outliers

Typically, they're aberrations or very distante values.
However, some outliers are really informative and explain what's happening in the real world!

Detect with
- Hostogram plots: `sns.histplot()`
- Box plots: `sns.boxplot()`
- Residual plots: differences between the real/actual target values and the model predictions
- IQR calculation

The **Inter-Quantile Rage** (IQR) is used to detect outliers statistically/numerically.

```python
import numpy as np
q25, q50, q75 = np.percentile(data['Unemployment'], [25, 50, 75])
iqr = q75 - q25
minV = q25 - 1.5*iqr
maxV = q75 + 1.5*iqr
[x for x in data['Unemployment'] if x > maxV]
```

The **computation of residuals** can be done in several ways:
- Simple difference between reality and model values
- Standardized: difference divided by standard error
- etc.

**Policies** or actions we can take with outliers:

- Remove them; but we might lose relevant information. Additionally, we lose the entire row.
- Assign/impute the mean, median, etc. But we might be losing relevant information
- Transform the variable: maybe after a `log` transformation we have no outliers!
- Predict what the value should be
	- Using similar observations
	- using regression
- Keep them; but the model might turn very skewed

Always consider the fact that outliers might be telling us part of the story of what's happening, and they might be very important.
