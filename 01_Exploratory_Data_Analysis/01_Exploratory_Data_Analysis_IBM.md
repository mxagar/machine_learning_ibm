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
	- 2.1 Retrieving Data: CSV, JSON, SQL, NoSQL, APIs
	- 2.2 Lab Notebooks: Retrieving Data - `./lab/01a_DEMO_Reading_Data.ipynb`
	- 2.3 Data Cleaning
		- Missing Data
		- Outliers
	- 2.4 Lab Notebooks: Data Cleaning - `Data_Cleaning_Lab.ipynb`
3. Exploratory Data Analysis and Feature Engineering (Week 3)
	- 3.1 Exploratory Data Analysis (EDA)
		- Samples
		- Visualizations: Matplotib
		- Grouping Data: `pandas.groupby()`, `searborn.pairplot()`, `seaborn.jointplot()`, `seaborn.FacetGrid()`
	- 3.2 Lab Notebooks: Exploratory Data Analysis (EDA) - `01c_LAB_EDA.ipynb`
	- 3.3 Lab Notebooks: Exploratory Data Analysis (EDA) - `EDA_Lab.ipynb`
	- 3.4 Feature Engineering
		- Variable/Feature Transformation
		- Create new Features
		- Variable Selection
		- Feature Encoding
		- Feature Scaling
	- 3.5 Lab Notebooks: Feature Engineering - `01d_DEMO_Feature_Engineering.ipynb`
		- Simple EDA: load dataset, `info()`, `describe()`, remove id columns
		- One-hot encoding of dummy variables: detect object-type variables and use `pd.get_dummies()`
		- Log-transformation of skewed variables
		- Check NA values; `fillna()`
		- Selection of a subset of variables
		- Pairplot: check multi-colinearity is not strong
		- Feature engineering
			- Polynomial features; manual or with `sklearn`
			- Interaction terms
			- If a categorical variable has many levels, remove levels with few `value_counts()`
			- Create deviation features: apply `groupby` to a categorical variable and compute deviation factors of another variable withing each group
	- 3.6 Lab Notebooks: Feature Engineering - `Feature_Engineering_Lab.ipynb`
		- - (Check missing values)
		- Rename doubled categories in categorical variables
		- One-hot encoding of categoricals
		- Replace categorical levels by numerical values
		- **Date and time transformations**: convert string to date, extract month & day & hour, encode time zone, extract day name. Many date and time transformations are done in several ways.
		- Selection of relevant features by hand
		- Correlations: matrix & correlation wrt. target
		- PCA: sparsity of information is estimated detecting the number of variables that account for 95% of the explained variance.
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

### 3.4 Feature Engineering

Typical feature engineering steps, before data modeling:

- Transformations
- Extract / create new features, more descriptive
- Variable / feature selection
- Feature encoding
- Feature scaling

The following subsections introduce the most important concepts. The next section contains code examples of them.

Some general notes:

- Always make a copy of the dataset before staring to mess up with it.
- Use the `pd.info()` function to know the shape, the missing values and the types!

#### Variable/Feature Transformation

Linear regression models assume that there is a linear relationship between the features and the target/outcome variable; therefore, we need to transform the feautures if that relationship is not given.

Linear regression models also assume that the residuals (difference between the target and the predictions) are normally distributed. If that is note the case, we can again use feature transformations, that lead to those normal residuals.

A common approach is to check the skew of the numerical variables; if it's larger than `0.75`, a transformation is applied:

```python
mask = df.dtypes == np.float
float_cols = df.columns[mask]
skew_limit = 0.75 
skew_vals = df[float_cols].skew()
skew_cols = list(skew_vals[abs(skew_vals) > 0.75].sort_values().index)
for col in skew_cols.index.values:
    if col == "SalePrice": # Target
        continue
    df[col] = df[col].apply(np.log1p)
```

Typical transformations:

- `y = b_0 + b_1 * x`: linear model, no transformation
- `y = b_0 + b_1 * log(x)`: logarithmic transformation
- `y = b_0 + b_1*x + b_2 * x^2`: polynomial transformation

Note that the model remains to be linear, but with transformed features.

```python
# log = ln(x)
# log1p = ln(1+x), to avoid 0
from numpy import log, 
# Boxcox: it finds the ideal way to transform a skewed distribution
# to a normal one
from scipy.stats import boxcox

sns.distplot(data) # skewed
log_data = np.log(data) # log
sns.distplot(log_data) # check if now non-skewed
```

The polynomial transoformation can have different degrees, so that the modelled curve becomes more sophisticated. We are creating new features.

```python
from sklearn.preprocessing import PolynomialFeatures

polyFeat = PolynomialFeatures(degree=2) # square
polyFeat = polyFeat.fit(X)
X_poly = polyFeat.transform(X)
```

#### Create new Features

New, more descriiptive features can be created as follows:

- Multiply different features if we suspect there might be an interaction
- Divide different features, if the division has a meaning
- Create deviation factors from the mean of a numeric variable in groups or categories of another categorical variable. See next section with the lab notebook.

#### Variable Selection

We need to select relevant variables for our model.

But how? It's not explained in the course. Solution: I took the following code piece from my forked repository

[deploying-machine-learning-models](https://github.com/mxagar/deploying-machine-learning-models)

```python
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

# Train feature selection model with Lasso regression
# Lasso: L1 regularized linear regression
sel_ = SelectFromModel(Lasso(alpha=0.001, random_state=0))
sel_.fit(X_train, y_train)
# List of selected columns
selected_feats = X_train.columns[(sel_.get_support())]
```

Additionally, consider:

- Use pairplots to check multi-colineearity; correlated features are not good.
- Check NA values, and drop them if necessary.

#### Feature Encoding

Encoding is often applied to categorical features that can non-numeric values:

- Nominal, unordered: red, green, blue, True/False
- Ordinal, ordered: high, medium, low

General note: applying `value_counts` is essential; if we find caegories that have very few counts compared to the rest, maybe we should group them in a category called `Other`? See lab notebook in the next section.

Common encoding approached:

- Binary: 0, 1
- One-hot encoding: each level or category of a categorical variable becomes a binary feature
- Ordinal encoding: convert ordinal categories to `0,1,2,...`; but be careful: we're assuming the distance from one level to the next is the same -- Is it really so? Maybe it's better applying one-hot encoding?

```python
# Binary or one-hot encoding
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from pandas import get_dummies

# Ordinal encoding
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OrdinalEncoder
```

#### Feature Scaling

In algorithms that use some kind of distance measure between the multi-dimensional data points, it is stringly suggested scaling the features to similar ranges; otherwise, the data point vectors can be really overproportionate in certain directions.

Common scaling approaches:

- Standard scaling: subtract the mean and divide by the standard deviation; features are converted to standard normal viariables.
- Mix-max scaling: a mapping with which the minimum value becomes 0, max becomes 1. This is senstive to outliers!
- Robust scaling: IQR range is mapped to `[0,1]`, i.e., percentiles `25%, 75%`; thus, the scaled values go out from the `[0,1]` range.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
```

### 3.5 Lab Notebooks: Feature Engineering - `01d_DEMO_Feature_Engineering.ipynb`

Peronsal note: This notebook is quite messy; only example transformations are shown and no feature engineering is carried out from beginning to end on a dataset. Additionally, new concepts are introduced, which were not mentioned in the lectures. **However, the processing done here is very interesting. See the code summary below.**

The notebook uses the [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

Important note: always make a copy of the original loaded dataset!

Overview of Contents:

- Simple EDA: load dataset, `info()`, `describe()`, remove id columns
- One-hot encoding of dummy variables: detect object-type variables and use `pd.get_dummies()`
- Log-transformation of skewed variables
- Check NA values; `fillna()`
- Selection of a subset of variables
- Pairplot: check multi-colinearity is not strong
- Feature engineering
	- Polynomial features; manual or with `sklearn`
	- Interaction terms
	- If a categorical variable has many levels, remove levels with few `value_counts()`
	- Create deviation features: apply `groupby` to a categorical variable and compute deviation factors of another variable withing each group

In the following, the summary of the code pieces is provided.

```python

### -- Simple EDA: load dataset, `info()`, `describe()`, remove id columns

# Cols, missing data, types
df.info()
# Always make a copy!
data = df.copy()
# Drop columns that have identifier values
df.drop(['PID','Order'], axis=1, inplace=True)

### --  One-hot encoding of dummy variables: detect object-type variables and use `pd.get_dummies()`

# Filtering by string categoricals
one_hot_encode_cols = df.dtypes[df.dtypes == np.object]  
one_hot_encode_cols = one_hot_encode_cols.index.tolist()
# Do the one hot encoding; but I understand a concat is required to keep them
df = pd.get_dummies(df, columns=one_hot_encode_cols, drop_first=True)

### --  Log-transformation of skewed variables

# Create a list of float colums (numerical) to check for skewing
mask = df.dtypes == np.float
float_cols = df.columns[mask]
# Threshols skewness: 0.75 is typical
skew_limit = 0.75 
skew_vals = df[float_cols].skew()
# Get list of skewed numerical variables
skew_vals[abs(skew_vals) > 0.75].sort_values()
skew_cols = list(skew_vals[abs(skew_vals) > 0.75].sort_values().index)
# Apply log1p
for col in skew_cols.index.values:
    if col == "SalePrice": # Target
        continue
    df[col] = df[col].apply(np.log1p)

### --  Check NA values; apply `fillna()`

df.isnull().sum().sort_values()
# ...

### --  Selection of a subset of variables

# We would select some features...
X = df.loc[:,['Lot Area', 'Overall Qual', 'Overall Cond', 
                      'Year Built', 'Year Remod/Add', 'Gr Liv Area', 
                      'Full Bath', 'Bedroom AbvGr', 'Fireplaces', 
                      'Garage Cars']]
y = df['SalePrice']

### --  Pairplot: check multi-colinearity is not strong

sns.pairplot(X, plot_kws=dict(alpha=.1, edgecolor='none'))

### --  Polynomial features; manual or with `sklearn`

# Manual
X2 = X.copy()
X2['OQ2'] = X2['Overall Qual'] ** 2
X2['GLA2'] = X2['Gr Liv Area'] ** 2

# With Scikit-Learn
X2 = X.copy()
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2)
features = ['Lot Area', 'Overall Qual']
pf.fit(df[features])
feat_array = pf.transform(X2[features])
# Get a dataframe with the new polynomial features
# But it needs to be concatenated to X2
poly_df = pd.DataFrame(feat_array, columns = pf.get_feature_names(input_features=features))
X2 = pd.concat([X2,poly_df])

### --  Interaction terms

X3 = X2.copy()

# Multiplicative interaction
X3['OQ_x_YB'] = X3['Overall Qual'] * X3['Year Built']
# Division interaction
X3['OQ_/_LA'] = X3['Overall Qual'] / X3['Lot Area']

### --  If a categorical variable has many levels, remove levels with few `value_counts()`

X4 = X3.copy()
nbh_counts = X4.Neighborhood.value_counts()
other_nbhs = list(nbh_counts[nbh_counts <= 8].index)
X4['Neighborhood'] = X4['Neighborhood'].replace(other_nbhs, 'Other')

### --  Create deviation features: apply `groupby` to a categorical variable and compute deviation factors of another variable within each group

def add_deviation_feature(X, feature, category):
    
    # temp groupby object
    category_gb = X.groupby(category)[feature]
    
    # create category means and standard deviations for each observation
    category_mean = category_gb.transform(lambda x: x.mean())
    category_std = category_gb.transform(lambda x: x.std())
    
    # compute stds from category mean for each feature value,
    # add to X as new feature
    deviation_feature = (X[feature] - category_mean) / category_std 
    X[feature + '_Dev_' + category] = deviation_feature 

X5 = X4.copy()
X5['House Style'] = df['House Style']
add_deviation_feature(X5, 'Year Built', 'House Style')
add_deviation_feature(X5, 'Overall Qual', 'Neighborhood')
```

### 3.6 Lab Notebooks: Feature Engineering - `Feature_Engineering_Lab.ipynb`

Interesting notebook in which the following steps related to feature engineering are carried out:

- (Check missing values)
- Rename doubled categories in categorical variables
- One-hot encoding of categoricals
- Replace categorical levels by numerical values
- **Date and time transformations**: convert string to date, extract month & day & hour, encode time zone, extract day name. Many date and time transformations are done in several ways.
- Selection of relevant features by hand
- Correlations: matrix & correlation wrt. target
- PCA: sparsity of information is estimated detecting the number of variables that account for 95% of the explained variance.

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv('data/airlines_data.csv')
data.head()

### -- Check missing values

data.isnull().sum()
data = data.fillna(method='ffill') # Hold last sample value

### -- Rename doubled categories in categorical variables

data['Airline'].unique().tolist() # check doubles
data['Airline'] = np.where(data['Airline']=='Jet Airways Business',
	'Jet Airways', data['Airline'])

### -- One-hot encoding of categoricals

# Note that if columns passed, dummy vavirables of them are created
# while keeping and returning the rest of the dataframe; i.e., no need for concat
data1 = pd.get_dummies(data=data, columns = ['Airline', 'Source', 'Destination'])

### -- Replace categorical levels by numerical values

data.replace({"non-stop":0,
	"1 stop":1,"2 stops":2,"3 stops":3,"4 stops":4},inplace=True)

### -- Date and time transformations

# Typical Duration value: '7h 25m'
# Convert to hours & minutes integers
duration = list(data['Duration'])
for i in range(len(duration)) :
    if len(duration[i].split()) != 2:
        if 'h' in duration[i] :
            duration[i] = duration[i].strip() + ' 0m'
        elif 'm' in duration[i] :
            duration[i] = '0h {}'.format(duration[i].strip())

dur_hours = []
dur_minutes = []  
 for i in range(len(duration)) :
    dur_hours.append(int(duration[i].split()[0][:-1]))
    dur_minutes.append(int(duration[i].split()[1][:-1]))
     
data['Duration_hours'] = dur_hours
data['Duration_minutes'] = dur_minutes
data.loc[:,'Duration_hours'] *= 60
data['Duration_Total_mins'] = data['Duration_hours']+data['Duration_minutes']

# Typical Dep_Time value: 05:50
# Extract hours and minutes
data["Dep_Hour"] = pd.to_datetime(data['Dep_Time']).dt.hour
data["Dep_Min"] = pd.to_datetime(data['Dep_Time']).dt.minute

# Create time zone / day time variable
data['dep_timezone'] = pd.cut(data.Dep_Hour,
	[0,6,12,18,24],
	labels=['Night','Morning','Afternoon','Evening'])
data['dep_timezone']

# Typical Date_of_Journey: 24/03/2019
# Extract day, month, year, day of week
data['Month']= pd.to_datetime(data["Date_of_Journey"], format="%d/%m/%Y").dt.month
data['Day']= pd.to_datetime(data["Date_of_Journey"], format="%d/%m/%Y").dt.day
data['Year']= pd.to_datetime(data["Date_of_Journey"], format="%d/%m/%Y").dt.year
data['day_of_week'] = pd.to_datetime(data['Date_of_Journey']).dt.day_name()

### -- Selection of relevant features by hand

data.columns # display all columns and select relevant ones
new_data = data.loc[:,['Total_Stops', 'Airline_Air Asia',
       'Airline_Air India', 'Airline_GoAir', ..., 'Dep_Hour',
       'Dep_Min', 'dep_timezone', 'Price']]

### -- Correlations: matrix & correlation wrt. target

plt.figure(figsize=(18,18))
sns.heatmap(new_data.corr(),annot=True,cmap='RdYlGn')

features = new_data.corr()['Price'].sort_values()
features.plot(kind='bar',figsize=(10,8))

### -- PCA

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(new_data.astype(np.float64))

# PCA with all components
pca = PCA(n_components = X.shape[1])
pca.fit_transform(x)

# Explained variance of each principal component
explained_variance = pca.explained_variance_ratio_
explained_variance

# Sum all explained variances until 95% is reached;
# how many components do we need?
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >=0.95) + 1

# Another way to answer the question above:
# pass varinace ratio float as n_components
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
X_reduced.shape[1]

```

## 4. Inferential Statistics and Hypothesis Testing (Week 4)

This section is quite bad. No real practical explanations given, in my opinion.

My notes are not introductory; I understand them because I have already worked on these things.

For inference and hypothesis testing, refer to my notes on the topic from the Coursera course:

[Statistics with Python](https://github.com/mxagar/statistics_with_python_coursera) / `02_Inference` / `lab`

My local folder:

`~/git_repositories/statistics_with_python_coursera`


### 4.1 Estimation, Inference, and Hypothesis Testing: Differences

- Estimation: for instance, we compute mean of a population. It is an estimate of the real mean, because we have used a sample.
- Inference: we put an accuracy on the estimation of our parameter, for instance the standard error. This implies either confidence intervals or statistical significances.
- Machine learning is very close to statistical estimation; however, we can also have estimations and accuracies.

#### Exploratory Data Analysis

In any case, the very first case consists in performing a good exploratory data analysis and visualizing the data. An example is given for the case of customer churn of a fictional telecom company. These information is plotted:

- Barplot of churn rate (`churn_value`) depending on payment type.
- Barplot of churn rate depending on customer duration in months (`pd.cut()` is used to put numerical values into buckets).
- Paiplot of numerical variables using as `hue = churn_value`.
- Jointplot with hexplot between monthly charge and customer duration in months

#### Parametric vs. Non-Parametric Statistics

In contrast to parametric statistics, in non-parametric statistics 

- in general, we make fewer assumptions
- and, in particular, the data don't need to belong to a distribution

Parametric models, in contrast to non-parametric ones, have analytic formular which depend on a limited number of parameters.

Example of parametric model: Normal distirbution; parameters: mean and standard deviation / variance.

The Maximum Likelihood Estimation is used to estimate the parameters of a model.

#### Commonly Used Distributions

- Uniform: the chances of rolling a number from a die
- Normal: popular because of the **Central Limit Theorem**: the sampling distirbution of the means is normal if we have enough samples and data. It's a distribution with very low likelihoods for extreme values; as such, it appears in natural and physical phenomena such as height and weight.
- **Log-normal: if we take the log of a log-normal, we have the normal distribution. It looks like a skewed normal distribution with longer tails. Very common!**
	- Examples: household income, etc.
- **Exponential**: most of the values closer to the left side.
	- Typical example: amount of time before the next event; the time between two people watching this video?
- **Poisson**: number of events that happen during a certain amount of time. Parameter: `lambda`, both average and variance.
	- Example: How many people are going to watch this video in the next ten minutes?

#### Frequentist vs. Bayesian Statistics

- Frequentists: repeated observations in the limit
	- Probabilities are modelled from the data; we perform many experiments.
- Bayesians: parameters are described as distributions
	- We have prior distirbutions or beliefs (educated guesses), which are updated with the data
	- After the update, we have the posterior distribution of the parameter

In both frameworks, the maths is the same; the interpretation and the process are different.

### 4.2 Hypothesis Testing

The videos of this section are not good; not very practical.

We have: 

- `H0`: null hypothesis; default specific value (e.g., mean) we see if we can reject to take the alternative. Considering `H0` is true, which is the probability that the data is in accordance to it? If the probability is low (e.g., lower than 5%), we reject the `H0` in favor of the `Ha`
- `Ha`: alternative hypothesis; taken as valid when the `H0` is rejected.

The `H0` defines the decision boundary in the frewuentist framework; in the Bayesian framework, we don't have boundaries, instead, we get updated posterior probabilities of `H0` and `Ha` and decide wwhich one is more likely.

#### Errors: Type I & Type II

- Accept `H0`, but it's wrong: Type I
- Accept `Ha` (i.e., reject `H0`), but it's wrong: Type II

![Type I and II Errors](type_1_2_errors.png)

- P(Type I error) = `alpha`, significance level, typically 0.05
- P(Type II error) = `beta`
- Power of a test = 1 - `beta`

The examples given are not good. I think the main idea here is that `alpha` and `beta` depend on the distirbutions of `H0` and `Ha`.

#### Significance Levels and P-Values

The **significance level** is the value below which we reject our null hypothesis: `alpha`; it's value depends on the importance of the Type I and II errors.

Example: we have a new medication which can heal some desease, but also can cause some side effects; the `alpha` for testing its healing effeicacy should be very low (e.g. `0.001`)! For other not that delicate cases, values between `0.01 - 0.05` are common.

Not choosing the `alpha` before computing the test statistic is p-hacking.

The **p-value** is the probability under null distirbution of seeing a result more extreme than what was observed. So we assume `H0` is true.

The **confience interval**: the values of the test statistic for which we accept the null.

Practical interpretation of `alpha = 0.05`: `1 - 0.05 = 0.95 = 95%` of observations are inside the region threshold, which is 2 standard deviations in the case of the normal distribution.

#### F-Statistic

The F value in regression is the result of a test where the null hypothesis is that all of the regression coefficients are equal to zero, i.e., there is no model that predicts better than the mean of all values.

We check the p-value of the F value/statistic: if it's below `alpha = 0.05`, we reject the null and consider the model valid.

Even thou the F statistic is computed in a different manner in ANOVA, the F distribution used seems to be the same: the F distribution is the ratio of two chi-square distributions with degrees of freedom df1 and df2.

#### Power and Sample Size

If we do many %5 significance tests (i.e., many group comparisons or multiple samplings) looking for a significant result, the chances of making at least one Type I error increase.

The probability of at least one Type I error is approx: `1 - (1-0.05)^num_tests`. This is roughly `0.05 x num_tests` if `num_tests < 10`.

Thus, the Bonferroni correction: we adjust `alpha`: `0.05 / num_tests`.

The Bonferroni correction decreases the Type I error, but increases the Type II error, i.e., we decrease the power of our test.

### 4.3 Lab Notebooks: Hypothesis Testing - `01e_DEMO_Hypothesis_Testing.ipynb`

The notebook is quite theoretical: the example of predicting 100 fair coin tosses is analyzed.

We toss a fair coin 100 times and claim we can predict its value above 56 of the times:

- `H0`: we cannot predict, our rate is 0.5
- `Ha`: we can predict, our rate is larger than 0.5

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats 
import math

# Binomial distirbution: n Bernoulli trials
# Bernuoulli distribution: distirbution for two categories with p and 1-p
from scipy.stats import binom

# 1) Test: Which is the probability of observing >56 considering H0 true?
# CDF: Cummulative density function: integral/area of the density function
# We need to do 1-CDF(tosses)
# because we're checking the prob of guessing those tosses or more
prob = 1 - binom.cdf(56, 100, 0.5)
print(str(round(prob*100, 1))+"%") # 9.7% < alpha = 5 %

# 2) Which is the critical prediction rate to reject H0?
# PPF: Percent point function:
# we insert the % of the CDF and get the variable to get that % 
print(binom.ppf(0.95,100,0.5)+1) # 59
```

In the notebook, it is further shown that increasing the sample size (1000 tosses instead of 100) increases the chance of getting significant results! That is basically because we reduce the variance, i.e., the standard error.

### 4.4 Correlation vs. Causation

Does it rain more on cooler days? We tend to associate cool with rain, but in some regions, warmth increases humidity, which leads to rain.

If two variables X and Y are correlated, X is useful for predict Y; the correlation might be due to different reasons:

- X causes Y
- Y causes X
- X and Y are both caused by something else (confounding)
- Y and Y aren't really related, we have a lucky sample which causes a spurious correlation

#### Confounding Variables

X and Y are correlated, but none of them causes the other, instead, both are caused by something else.

Examples:

- Car accidents and people called John are correlated; reason: both correlated with population size: the larger the population, the larger the number of car accidents and people called John.
- Amount of ice-cream and drownings are correlated; reason: both increased by temperature: more people eat ice-cream and go swimming.

#### Spurios Correlations

Correlations that occur by chance.

### Personal Notes

This section is bad.

For inference and hypothesis testing, refer to my notes on the topic from the Coursera course:

[Statistics with Python](https://github.com/mxagar/statistics_with_python_coursera) / `02_Inference` / `lab`

My local folder:

`~/git_repositories/statistics_with_python_coursera`

