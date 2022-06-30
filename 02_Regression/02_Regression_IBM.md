# Supervised Machine Learning: Regression

These are my notes and the code of the [IBM Machine Learning Professional Certificate](https://www.coursera.org/professional-certificates/ibm-machine-learning) offered by IBM & Coursera.

The Specialization is divided in 6 courses, and each of them has its own folder with its guide & notebooks:

1. [Exploratory Data Analysis for Machine Learning](https://www.coursera.org/learn/ibm-exploratory-data-analysis-for-machine-learning?specialization=ibm-machine-learning)
2. [Supervised Machine Learning: Regression](https://www.coursera.org/learn/supervised-machine-learning-regression?specialization=ibm-machine-learning)
3. [Supervised Machine Learning: Classification](https://www.coursera.org/learn/supervised-machine-learning-classification?specialization=ibm-machine-learning)
4. [Unsupervised Machine Learning](https://www.coursera.org/learn/ibm-unsupervised-machine-learning?specialization=ibm-machine-learning)
5. [Deep Learning and Reinforcement Learning](https://www.coursera.org/learn/deep-learning-reinforcement-learning?specialization=ibm-machine-learning)
6. [Specialized Models: Time Series and Survival Analysis](https://www.coursera.org/learn/time-series-survival-analysis?specialization=ibm-machine-learning)

This file focuses on the **second course: Supervised Machine Learning: Regression**

Mikel Sagardia, 2022.  
No guarantees

## Overview of Contents

1. [Introduction to Supervised Machine Learning](#1.-Introduction-to-Supervised-Machine-Learning)
	- 1.1 Interpretation vs. Prediction
2. [Linear Regression](#2.-Linear-Regression)
	- 2.1 Model Definition
	- 2.2 Model Evaluation: R2
	- 2.3 Python Code with Scikit-Learn
	- 2.4 Python Lab: `02a_LAB_Transforming_Target.ipynb`
3. [Training and Test Splits](#3.-Training-and-Test-Splits)
4. [Cross-Validation](#4.-Cross-Validation)
5. [Polynomial Regression](#5.-Polynomial-Regression)

## 1. Introduction to Supervised Machine Learning

A model is a small thing that captures a big thing; as such, it reduces the complexity while capturing the feeatures we are insterested in.

We distinguish between:

- Supervised learning: the data is labelled with real outcome
- Unsupervised learning: the data is not labelled, instead, we find structure in it.
- Semi-supervised learning: the data is sometimes labelled and sometimes it is not.

The model for any supervised learning:

`y_p = f(Omega, X)`

- `y_p`: predicted outcome, in contrast to `y`, real outcome
- `f`: model function
- `Omega`: model parameters, learned; we say that the model is fit to the data.
- `X`: past data-points x features
- Hyperparameters: any para,eger which is not learned, e.g., whether we have an intercept or not.

Within supervised learning we distinguish:

- Regression: numeric, continuous outcome/target.
- Classification: categorical outcome/target.

In order to learn, a loss function is defined, which evaluates the difference between the real target `y` and the predicted `y_p`:

`J(y,y_p)`

In this course, we're going to work on a **housing dataset**; we're going to apply regression on house features to predict the prices.

### 1.1 Interpretation vs. Prediction

In some cases, the primary goal of training a models is not the prediction of the target given new data points, but the interpretation of the model.

When we seek interpretation, we focus on the parameters `Omega`: we want parameters that are clearly **interpretable**. For that interpretation we get the **feature importances** and often plot them as a bar plot.

Examples of cases in which model interpretation is important:

- x: Customer demographics, y: sales data; Omega: examine to understand loyalty by segment.
- Safety features which prevent car accidents?

In contrast, when prediction is sought, the interpretability of the model is not that important. We want to predict the targets as accurately as possible. In that case the loss function and the plot `y` vs. `y_p` are more important.

Thus:

- Interpretation: feature impostances in focus
- Prediction: actual vs. predicted output in focus

Usually a balance is desired between interpretability and predictability. Also, note that not all models are equally interpretable:

- Linear regression: very interpretable.
- Deep learning: not very interpretable.

## 2. Linear Regression

Overall, I'd say that the explanations are not as good as in the course by Andrew Ng: Machine Learning. I suggest looking at the linear regression module of that course to get theoretical details -- my notes: [machine_learning_coursera](https://github.com/mxagar/machine_learning_coursera).

The added value of the current IBM section is that it is shown how linear regression is done with available libraries professionally.

### 2.1 Model Definition

Example used: Predict Box Office Revenue of a movie given its Budget.

![Linear regression example](./pics/linear_regression_example.png)

The coefficients/parameters are found by minimizing the cost function = sum of all distances from the data points toa parametrized line.

![Cost function](./pics/cost_function.png)

### 2.2 Model Evaluation: R2

**Coefficient of determination, R2**: It measures the percentage of variance that is explained by the model. Interesting insight: ratio between

- Sum of squared Error (SSE) = `sum((y_pred - y)^2)`
- Total Sum of Squares (TSS) = Variance of observed `y` = `sum((mean(y) - y)^2)`

`R2 = 1 - (SSE/TSS)`

![Coefficient of determination, R2](./pics/r2.png)

Notes:

- R2 is the ration between the variance captured by the model and the total variance.
- If we add features but these have no predictive power, R2 will not improve.
- SSE is used for minimizing the cost function but TSS is the total observed variance of the target in the dataset, which is always constant!


### 2.3 Python Code with Scikit-Learn

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
```

### 2.4 Python Lab: `02a_LAB_Transforming_Target.ipynb`

This notebook contains the following: 

- Different transformations of the target are tested; although it is not compulsory having a normally distirbuted target, it improves the model performance generally.
	- Log
	- Square root
	- Box cox = generalized power transformation, for which the optimum power coefficient is found: `boxcox = (y^lambda - 1)/lambda` 
- The linear model is loaded.
- Polynomial features are computed.
- Train/test split + scaling are applied.
- The model is fit.
- The inverse transformation is applied to the predicted values.
- Another fit is done with the untransformed target; the R2 is lower.

There is a helper script `helper.py` which loads the boston housing dataset from Scikit-Learn.

In the following, a summary of the most important parts in the notebook:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.simplefilter("ignore")

### -- 1. Load the dataset

from helper import boston_dataframe
boston_data, description = boston_dataframe(description=True)
# from sklearn.datasets import load_boston
# boston = load_boston()
# ...

boston_data.shape # (506, 14)
# Target: MEDV

print(description) # all variables described

### -- 2. Target transformation

# Even though it is not necessary to have a normally distributed target
# having it so often improves the R2 of the model.
# We can check the normality of a variable in two ways:
# - visually: hist(), QQ-plot
# - with normality checks, e.g., D'Agostino

# Normality check: D'Agostino
# if p-value > 0.05: normal;
# the larger the p-value, the larger the probability of normality
from scipy.stats.mstats import normaltest # D'Agostino K^2 Test
normaltest(boston_data.MEDV.values) # pvalue=1.7583188871696095e-20

# Square root transformation
sqrt_medv = np.sqrt(boston_data.MEDV)
plt.hist(sqrt_medv)
normaltest(sqrt_medv) # pvalue=3.558645701429252e-05

# Square root transformation
log_medv = np.log(boston_data.MEDV)
plt.hist(log_medv)
normaltest(log_medv) # pvalue=0.00018245472768345196

# Box-Cox transformation: Generalized power transformation
# boxcox = (y^lambda - 1)/lambda
# It requires y > 0; else apply y+y_min or use Yeo-Johnson
# However, Box-Cox seems easier to interpret/explain.
# Note: always save the lambda!
from scipy.stats import boxcox
bc_result = boxcox(boston_data.MEDV)
boxcox_medv = bc_result[0]
lmbd = bc_result[1]
plt.hist(boxcox_medv)
normaltest(boxcox_medv) # pvalue=0.104... NORMAL!

### -- 3. Model Fitting

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (StandardScaler, 
                                   PolynomialFeatures)

lr = LinearRegression()

y_col = "MEDV"
X = boston_data.drop(y_col, axis=1)
y = boston_data[y_col]s
X.shape # (506, 13)

# We create 2nd degree variables of the existing ones
# Note: bias will be added by the regression model
pf = PolynomialFeatures(degree=2, include_bias=False)
X_pf = pf.fit_transform(X)
X_pf.shape # (506, 104)
pf.get_feature_names_out() # get all feature names after the polynomial computation

# Train/test split
# ALWAYS set a random state!
X_train, X_test, y_train, y_test = train_test_split(X_pf, y, test_size=0.3, 
                                                    random_state=72018)

# Scale: after the PolynomialFeatures
s = StandardScaler()
X_train_s = s.fit_transform(X_train)

# Transform the target to be normal
bc_result = boxcox(y_train)
y_train_bc = bc_result2[0]
lmbd = bc_result2[1]

# Fit/Train
lr.fit(X_train_s, y_train_bc)

### -- 4. Evaluation

# Predict
X_test_s = s.transform(X_test)
y_pred_bc = lr.predict(X_test_s)

# Untransform the predicted y
# We need the inverse of the transformation function and its parameters!
from scipy.special import inv_boxcox
y_pred = inv_boxcox(y_pred_bc,lmbd)
r2_score(y_pred,y_test) # 0.8794001850404838

# What if we would have not used the box-cox transformation?
# The R2 would have been worse!
lr = LinearRegression()
lr.fit(X_train_s,y_train)
lr_pred = lr.predict(X_test_s)
r2_score(lr_pred,y_test) # 0.8555202098064152

```

## 3. Training and Test Splits



## 4. Cross-Validation



## 5. Polynomial Regression


