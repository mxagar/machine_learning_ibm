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

### 2.2 Lab Notebooks: Retrieving Data - `./lab/01a_DEMO_Reading_Data.ipynb`

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

### 2.4 Lab Notebooks: Data Cleaning - `Data_Cleaning_Lab.ipynb`

The notebook `Data_Cleaning_Lab.ipynb` is a nice summary of the most important steps to perform when data cleaning is done in a research environment. However, if we want to prepare everything for a development/production envvironment, the code needs to be modified...

In the following, a summary of the contents:
- Import relevant libraries
- Get data: `info()`, `describe()`
- Categorical: `value_counts()`
- Numerical: correlations with target
- Target: skewness & transformations: `np.log()`, `np.sqrt()`, `stats.boxcox()`
- Duplicate rows: detect & drop
- Missing values
	- Detect and plot (`bar`)
	- Drop or impute with mean/median
- Feature scaling: `MinMaxScaler()`, `StandardScaler()`
- Outliers
	- Detect visually: `boxplot()`, `scatter()`
	- Compute `stats.zscore()` to check how many std. devs. from mean
	- Consider applying transformations: `np.log()`, `np.sqrt()`, `stats.boxcox()`
	- Numerically: `sort_values(by, ascending=False)`
	- Drop if we see that they do not represent reality

In the following, a summary of the code:

```python

### -- Import relevant libraries

import pandas as pd
import numpy as np 

import seaborn as sns 
import matplotlib.pylab as plt
%matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import norm
from scipy import stats

### -- Get data: `info()`, `describe()`

housing.to_csv('data/housing.csv',sep=',',header=True,index=False)
housing = pd.read_csv('data/housing.csv')

housing.head(10)
housing.info()
housing["SalePrice"].describe()

### -- Categorical: `value_counts()`

# Get counts of the levels of a categorcial variable
housing["Sale Condition"].value_counts()

### -- Numerical: correlations with target

# Get correlations between target & numerical dependent variables
# Select high correlations and order them descending
hous_num = housing.select_dtypes(include = ['float64', 'int64'])
hous_num_corr = hous_num.corr()['SalePrice'][:-1] # -1 means that the latest row is SalePrice
top_features = hous_num_corr[abs(hous_num_corr) > 0.5].sort_values(ascending=False) #displays pearsons correlation coefficient greater than 0.5
print("There is {} strongly correlated values with SalePrice:\n{}".format(len(top_features), top_features))

# Pair-scatterplots between target and numerical variables
# for visual inspection
for i in range(0, len(hous_num.columns), 5):
    sns.pairplot(data=hous_num, x_vars=hous_num.columns[i:i+5], y_vars=['SalePrice'])

### -- Target: skewness & transformations: `np.log()`, `np.sqrt()`, `stats.boxcox()`

# Distribution of target: it should be bell-shaped, as standardized as possible
sp_untransformed = sns.distplot(housing['SalePrice'])
# Compute skewness: if it deviates much from 0, we have tails
print("Skewness: %f" % housing['SalePrice'].skew())

# Try: apply log transformation and check distirbution and skewness again
log_transformed = np.log(housing['SalePrice'])
print("Skewness: %f" % (log_transformed).skew())

# Try: apply sqrt transformation and check distirbution and skewness again
sqrt_transformed = np.sqrt(housing['SalePrice'])
print("Skewness: %f" % sqrt_transformed.skew())
sns.distplot(sqrt_transformed)

# Try: apply boxcox transformation and check distirbution and skewness again
boxcox_transformed = pd.Series(stats.boxcox(housing['SalePrice'])[0])
print("Skewness: %f" % boxcox_transformed.skew())
sns.distplot(boxcox_transformed)

### -- Duplicate rows: detect & drop

# Get duplicated rows
# housing.duplicated(['PID']) -> False, False, ...
duplicate = housing[housing.duplicated(['PID'])]

# Drop duplicates
dup_removed = housing.drop_duplicates()

# Check that all indices are unique
housing.index.is_unique

### -- Missing values

# Detect missing values, sort them ascending, plot
total = housing.isnull().sum().sort_values(ascending=False)
total_select = total.head(20)
total_select.plot(kind="bar", figsize = (8,6), fontsize = 10)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Count", fontsize = 20)
plt.title("Total Missing Values", fontsize = 20)

# Detect missing values, sort them ascending, plot
# I would drop all features/columns with more than 15% of missing values
# and impute with the median/mean the rest; maybe impute with median the ones with =<%5
# predict the ones with 5-15%
total = housing.isnull().sum().sort_values(ascending=False)
total_select = total/housing.shape[0] # Normalize to get percentages
total_select = total_select.head(20) # Select the 20 most relevant feeatures
total_select.plot(kind="bar", figsize = (8,6), fontsize = 10)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Count", fontsize = 20)
plt.title("Total Missing Values", fontsize = 20)

# All rows with NA in column "Lot Frontage" dropped, but not inplace!
housing.dropna(subset=["Lot Frontage"])

# Entire column "Lot Frontage" dropped, but not inplace!
housing.drop("Lot Frontage", axis=1)

# Compute median of a column/feature
median = housing["Lot Frontage"].median()
# Replace/impute NA values with median
housing["Lot Frontage"].fillna(median, inplace = True)

# Replace/impute NA values with mean
mean = housing["Mas Vnr Area"].mean()
housing["Mas Vnr Area"].fillna(mean, inplace = True)

### -- Feature scaling: `MinMaxScaler()`, `StandardScaler()`

# Fit scaler and transform data: normalize
# MinMaxScaler: divide each column/feature with (max-min)
# However, I think it is better to instantiate an object and fit() & transform()
# separately; then, we need to save the scaler as pickle or another serialized object!
norm_data = MinMaxScaler().fit_transform(hous_num)

# StandardScaler: (value - mea) / std
scaled_data = StandardScaler().fit_transform(hous_num)

# The Scikit-Learn scalers seem to work with ND arrays
# so columns need to be passed as arrays of arrays?
scaled_sprice = StandardScaler().fit_transform(housing['SalePrice'][:,np.newaxis]) 

### -- Outliers

# Box plot: detect outliers that are outside the 1.5*IQR
# Keeping or removing them depends on our understanding of the data
sns.boxplot(x=housing['Lot Area'])

# Also consider transformations! Maybe outlibers disappear after a log transformation
sns.boxplot(x=np.log(housing['Lot Area']))

# Also perform bi-variate scatter-plots: feature vs target
# see if some data-points are indeed outside of what is expected
# Does it make more sense to perform that with features with the highest correlations?
price_area = housing.plot.scatter(x='Gr Liv Area', y='SalePrice')

# In the scatterplot, the datapoints with the 2 largest values for 'Gr Liv Area' seem to be outliers
# We get their rows
housing.sort_values(by = 'Gr Liv Area', ascending = False)[:2]

# Manually removing using the index
outliers_dropped = housing.drop(housing.index[[1499,2181]])

# Detect outliers in the feature/column 'Lot Area':
# box-plot (univariate) & scatter-plot (with target)
# Spot possible outliears visually
sns.boxplot(x=housing['Lot Area'])

price_lot = housing.plot.scatter(x='Lot Area', y='SalePrice')

# Z-score statistic: how many standard deviations away from the mean
# Recall the empricial rule: +-1 std 67%, +- 2 std 95%, +- 3 std 99.7%
# Often the values beyond +- 3std or Z-score are considered outliers
# BUT: maybe data is not normally distirbuted, instead we have a fat-tails distribution
# and the high z-scores are points at fat tails that explain the reality!!
housing['Lot_Area_Stats'] = stats.zscore(housing['Lot Area'])

# Describe provides with quantiles of features
housing[['Lot Area','Lot_Area_Stats']].describe().round(3)

# Get datapoints sorted by Lot Area
# After the visual inspection of the plots,
# the last (larger) ones are candidates to be removed
# They are also >10 std. deviations away from the center/mean
housing.sort_values(by = 'Lot Area', ascending = False)[:5]

```

## 3. Exploratory Data Analysis and Feature Engineering (Week 3)

### 3.1 Exploratory Data Analysis (EDA)

We want to summarize the data: with number and diagrams, to:

- check whether more data is needed
- identify patterns
- etc.

We use:

- Summary statistics: avg., median, min, mx, correlations
- Viisualizations: histograms, scatter plots, box plots

In this section, the iris dataset is used, which can be loaded as follows

```python
import seaborn as sns
iris = sns.load_dataset('iris')
```

The iris dataset has the following dataframe:

```
(index) | sepal_length | sepal_width | petal_length | petal_width | species
species = {'setosa', 'versicolor', 'virginica'}
```

#### Samples

For larger datasets, we often perform samplings, i.e., we take samples; with that:

- we can have faster computations
- we can train on random samples
- we can control the over/underreppresentation of given groups

```python
# Pandas sample of n=5 rows/data-points
# replace=False (default): one row appears only once
sample = data.sample(n=5, replace=False)
```

#### Visualizations: Matplotib

Libraries used: Matplotlib. Pandas, Seaborn.

Matplotlib:

```python
import matplotlib.pyplot as plot
%matplotlib inline

# Scatterplot 1
plt.plot(data.sepal_length, data.sepal_width, ls='', marker='o', label='sepal')

# Scatterplot 2
# They are plotted in same diagram, if one after the other
plt.plot(data.petal_length, data.petal_width, ls='', marker='o', label='petal')

# Histogram
plt.hist(data.sepal_length, bins=25)

# Subplots
fig, ax = plt.subplots()
# Bars, horizontal
ax.barh(np.arange(10), data.sepal_width.iloc[:10])
# Set position of ticks and labels
ax.set_yticks(...)
ax.set_yticklabels(...)
ax.set(xlabel='...', ...)
```

#### Grouping Data: `pandas.groupby()`, `searborn.pairplot()`, `seaborn.jointplot()`, `seaborn.FacetGrid()`

```python
# GROUP BY
# When grouping by a column/field,
# we need to apply the an aggregate function
data.groupby('species').mean()

# PAIRPLOT
# All variables plotted against each other: scatterplots, histograms
# Hue: separate/group by categories
sns.pairplot(data, hue='species', size=3)

# JOINTPLOT
# Two variables plotted; type of scatterplot scecified + density histograms
sns.jointplot(x=data['sepal_length'],y=iris['sepal_width'],kind='hex')

# FACETGRID: map plot types to a grid of plots
# 1: specify dataset and col groups/levels
plot = sns.FacetGrid(data, col='species', margin_titles=True)
# 2: which variable to plot in cols, and which plot type (hist)
plot.map(plt.hist, 'sepal_width', color='green')
```

### 3.2 Lab Notebooks: Exploratory Data Analysis (EDA) - `01c_LAB_EDA.ipynb`

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('data/iris_dataset.csv')

# Number of rows
print(data.shape[0])
# Column names
print(data.columns.tolist())
# Data types
data.info()

# Number of species
data.species.value_counts()

# Quantiles
data.describe()

# Renaming: columns or categories
data['fam'] = data['species']
data = data.rename(columns={'fam': 'family'})
data['family'] = data['family'].replace({'setosa':'iris-setosa',
										 'virginca':'iris-virginica',
										 'versicolor':'iris-versicolor'})
# Undo example
data.drop('family',axis=1, inplace=True)

# Mean of each variable by species
data.groupby('species').median()
# Multiple functions
data.groupby('species').agg([np.mean, np.median])

# Scatterplot 
ax = plt.axes()
ax.scatter(data.sepal_length, data.sepal_width)
# Label the axes
ax.set(xlabel='Sepal Length (cm)',
       ylabel='Sepal Width (cm)',
       title='Sepal Length vs Width');

# Histogram
ax = plt.axes()
ax.hist(data.petal_length, bins=25);
ax.set(xlabel='Petal Length (cm)', 
       ylabel='Frequency',
       title='Distribution of Petal Lengths');

# Histograms overlapped: Pandas visualization
# A histogram for each numerical variable
sns.set_context('notebook')
ax = data.plot.hist(bins=25, alpha=0.5)
ax.set_xlabel('Size (cm)');

# Boxplot with Pandas:
# A boxplot for each numerical variable
# Hue specified with 'by'
data.boxplot(by='species');

# Pairplot
sns.set_context('talk')
sns.pairplot(data, hue='species');

```

### 3.3 Lab Notebooks: Exploratory Data Analysis (EDA) - `EDA_Lab.ipynb`

The notebook has two main parts: in teh first basic EDA is performed, using basically similar functions as in the previous notebook; in the second, the library `plotly.express` is used for nice interactive plots, showcasing

- Interactive line plots
- Animated bar plots
- Chloropeth: maps

In the following, a summary of interesting commands of the first part is provided:

```python
# Split the content of GEO to be two columns
# .str accesses the string
# n=1 performs 1 split
# expand=True returns a dataframe
data[['City', 'Province']] = data['GEO'].str.split(',', n=1, expand=True)

# Date handling
data['DATE'] = pd.to_datetime(data['DATE'], format='%b-%y')
data['Month'] = data['DATE'].dt.month_name().str.slice(stop=3)
data['Year'] = data['DATE'].dt.year

# Multiple filtering
mult_loc = data[(data['GEO'] == "Toronto, Ontario") | (data['GEO'] == "Edmonton, Alberta")]
cities = ['Calgary', 'Toronto', 'Edmonton']
CTE = data[data.City.isin(cities)]

# Group ba with multiple columns/fetaures
data.groupby(['Year', 'City'])['VALUE'].median()
```