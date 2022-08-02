# Supervised Machine Learning: Regression

These are my notes and the code of the [IBM Machine Learning Professional Certificate](https://www.coursera.org/professional-certificates/ibm-machine-learning) offered by IBM & Coursera.

The Specialization is divided in 6 courses, and each of them has its own folder with its guide & notebooks:

1. [Exploratory Data Analysis for Machine Learning](https://www.coursera.org/learn/ibm-exploratory-data-analysis-for-machine-learning?specialization=ibm-machine-learning)
2. [Supervised Machine Learning: Regression](https://www.coursera.org/learn/supervised-machine-learning-regression?specialization=ibm-machine-learning)
3. [Supervised Machine Learning: Classification](https://www.coursera.org/learn/supervised-machine-learning-classification?specialization=ibm-machine-learning)
4. [Unsupervised Machine Learning](https://www.coursera.org/learn/ibm-unsupervised-machine-learning?specialization=ibm-machine-learning)
5. [Deep Learning and Reinforcement Learning](https://www.coursera.org/learn/deep-learning-reinforcement-learning?specialization=ibm-machine-learning)
6. [Specialized Models: Time Series and Survival Analysis](https://www.coursera.org/learn/time-series-survival-analysis?specialization=ibm-machine-learning)

This file focuses on the **third course: Supervised Machine Learning: Classification**

Mikel Sagardia, 2022.  
No guarantees

## Overview of Contents

1. [Logistic Regression (Week 1)](#1.-Logistic-Regression)

## 1. Logistic Regression

### 1.1 What is Classification?

Supervised learning consists of Regression and Classification. Regression problems provide a continuous output. Classification problems provide a category output; some examples:

- Detecting fraudulent transactions
- Customer churn or not
- Event attendance
- Network load
- Loan default

There are may classification techniques:

- Logistic regression
- K-Nearest Neighbors
- Support Vector Machines
- Neural Networks
- Decision Trees
- Random Forests
- Boosting
- Ensemble Models

However, **each of these models can be really both for regression and classification!!**

### 1.2 Logistic Regression

We could treat a binary classification problem as a regression problem if we plot our continuous predictor in the `x` axis and the binary target in the `y` axis. Then, we fit the line in the data-points and set the classification threshold in the `x` value which yields `y = 0.5`. Example with customer churn:

![Linear regression for classification](./pics/linear_regression_classification.png)

The problem with approach is that the line tilts towards the regions with high point density, moving the threshold with it.

A solution to that is using the **sigmoid function**, which shrinks the line to the extremes. The sigmoid takes the full range of real numbers and maps them all to `(0,1)`.

```
sigmoid(x) = 1 / (1 + exp(-x))
```

![Logistic regression model](./pics/logistic_regression_model.png)

The resulting model is the **logistic regression** model, which predicts the probability of `x` belonging to a class.:

```
y = p(x) = sigmoid(beta_0 + beta_1 * x) = sigmoid(B * X)
```

Note that the **log odds ratio** can be computed as follows:

```
log(p(x) / (1 - p(x))) = beta_0 + beta_1 * x
```

![Log-odds ratio](./pics/log_odds_ratio.png)

**That is an important insight: each model coefficient measures the effect of its associated feature on the log-odds ratio.** Linear regression has an equivalent interpretation, but with respect to the target value.

We can have higher dimensions for the `x` variable, such that the threshold becomes a linear boundary or a hyperplane.

### 1.3 Classification with Multiple Classes

Muti-class classification can be performed with the *one-vs-all* technique: a binary classification model is composed for each class versus the rest of the classes. This way, each point in feature space gets one probability for each of the classes, and we pick the largest one to create the regions and boundaries.

![Multi-class classification](./pics/multi_class_classification.png)

Another approach consists in using **multinomial models**. These models are generalized models which a probability distribution over all classes, based on the logits or exponentiated log-odds calculated for each class (usually more than two).

Note: Multinomial is not Multivariate:

- Multivariate distribution: the feature is a vector.
- Multinomial: the target is a vector, i.e., we have a generalized binomial distirbution.

### 1.4 Logistic Regression in Python with Scikit-Learn

[Scikit-Learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

```python
# Imports
from sklearn.linear_model import LogisticRegression

# Instantiate model
# We specify the regularization parameters:
# L2 regularization
# c = 1/lambda, inverse of the regularization strength,
# i.e., the bigger c, the lower the regularization.
LR = LogisticRegression(penalty='l2', c=10.0)

# Fit and predict
LR = LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)

# Coefficients
# Also, consider using statsmodels
# because we get confidence intervals and significances
# for the coefficients
# Note that larger coefficients denote larger influence
# in the class outcome
LR.coef_

# Tune regularization parameters with the cross-validation model
LogisticRegressionCV
```

### 1.5 Classification Error Metrics: Confusion Matrix, Accuracy, Specificity, Precision, and Recall

Accuracy is a really bad metric for classification if we have imbalanced classes (which is often the case). Example: if we want to predict cancer and our dataset consists of 1% sick and 99% healthy; a simple model which always predicts "healthy" is wrong, but still accurate 99% of the time.

The basis for computing classification metrics is the **confusion matrix**, which is the matrix that counts the cases for `real (True / False) x predicted (True / False)`.

In that matrix:

- False positives are the **Type I** error.
- False negatives are the **Type II** error; in the case of sickness prediction, we want to significantly minimize this type of error.

From the matrix, we compute the most common error metrics:

- Accuracy: diagonal / all 4 cells
- Precision (of Predicted Positives) = TP / (TP + FP) 
- Recall or Sensitivity (wrt. Real Positives) = TP / (TP + FN)
- Specificity: Precision for Negatives = TN / (FP + TN)
- F1: harmonic mean between precision and recall; it is a nice trade-off between precision and recall, thus, a metric which is recommend by default: `F1 = 2 *(P*R)/(P+R)`

![Error Measurements with the Confusion Matrix](./pics/error_measurements.png)

### 1.6 ROC and Precision-Recall Curves

**ROC = Receiver Operating Characteristic.**

If we take a model, vary the classification threshold and plot its true positive and false positive rates pairs, we get the ROC curve: [Receiver Operating Characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic).

Note that:

- True positive rate = Sensitivity = Recall
- False positive rate = 1 - Specificity

Now, we can compute the **Area Under the Curve (AUC)**; its value has the following interpretation:

- ROC-AUC = 0.5: the model performs like random guessing.
- The larger ROC-AUC the better performs the model.
- The curve will be convex in practice.

![ROC Area Under the Curve (AUC)](./pics/roc.png)

Similarly, we can compute another curve, which is better suited **for umbalanced classes: Precision-Recall Curve**

![Precision-Recall curve](./pics/precision_recall_curve.png)

The Precision-Recall curve measures the Precision-Recall values for varied thresholds; it is usually a descending curve.

Ultimately, the application or use-case we're dealing with is essential to choosing the correct metric: F1, ROC-AUC, etc. We need to consider which cases we want to avoid because they have a too high cost.

### 1.7 Multi-Class Metrics

If we have a confusion matrix with several classes (i.e., larger than `2x2`), we can compute the accuracy with the values in the diagonal, but the other metrics are not really generalizable for the complete dataset.

However, we can compute each one of them for each class, i.e., taking the *one-vs-all* approach. Then, a weighted average can be computed, i.e., weighting with the ratio of each class.

### 1.8 Metrics in Python with Scikit-Learn

```python
# Accuracy
from sklearn.metrics import accuracy_score

# Compute the accuracy
accuracy_value = accuracy_score(y_test, y_pred)

# Other error metrics
from sklearn.metrics import (precision_score, recall_score,
          f1_score, roc_auc_score,
          confusion_matrix, roc_curve,
          precision_recall_curve)

```

### 1.9 Python Lab: Human Activity

In this notebook, the [Human Activity Recognition Using Smartphones Data Set ](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML241ENSkillsNetwork31576874-2022-01-01) is used: users carried out a smartphone with an inertial sensor and carried out Activities of Daily Living (ADL); these activities are annotated/labeled as

- `WALKING`,
- `WALKING UPSTAIRS`,
- `WALKING DOWNSTAIRS`,
- `SITTING`, 
- `STANDING`,
- and `LAYING`.

Dataset records contain:

- Triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration.
- Triaxial Angular velocity from the gyroscope.
- Many time and frequency domain variables: moving average, kurtosis, etc.
- The `Activity` label with the 6 classes listed above.

Altogether we have a 561-feature vector with values already scaled to `[-1,1]`. No cleaning needs to be carried out and all variables are continuous; the only pre-processing consists in converting the class names into integers.

The notebook is interesting because it shows how to deal with different models and compute classifications metrics in a multi-class example (class ratios need to be maintained in splits).

Steps:

1. Load dataset, inspect and prepare it
2. Check correlations
3. Split dataset maintaining class ratios: StratifiedShuffleSplit
4. Define and fit logistic regression models with and without regularization
5. Compare the magnitude of the coefficients of each model in a multi-level dataframe
6. Predict the probabilities and the classes of the test split for each model
7. Compute the metrics for each model: precision, recall, f1, accuracy, roc-auc, confusion matrix

In the following, the code of the notebook:

```python
import seaborn as sns, pandas as pd, numpy as np

# Display all the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)

### -- 1. Load dataset, inspect and prepare it

#data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/Human_Activity_Recognition_Using_Smartphones_Data.csv", sep=',')
data = pd.read_csv("data/Human_Activity_Recognition_Using_Smartphones_Data.csv")

data.dtypes.value_counts()

# The min for every single feature column is -1
data.iloc[:, :-1].min().value_counts()
# The max for every single feature column is 1
data.iloc[:, :-1].max().value_counts()

# Target column: always do value_counts
# to see how balanced the classes are.
# They are quite balanced.
data.Activity.value_counts()

# Classification problems require passing label-encoded target values,
# one-hot encoded (sparse) values are not accepted.
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Activity'] = le.fit_transform(data.Activity)
data['Activity'].sample(5)

### -- 2. Check correlations

# Calculate the correlation values between the feature values
feature_cols = data.columns[:-1]
corr_values = data[feature_cols].corr()

# Calculate the correlation values between the feature values
feature_cols = data.columns[:-1]
corr_values = data[feature_cols].corr()

# Simplify by emptying all the data below the diagonal.
# tril_indices_from returns a tuple of 2 arrays:
# the arrays contain the indices of the diagonal + lower triangle of the matrix:
# ([0,1,...],[0,0,...])
tril_index = np.tril_indices_from(corr_values)

# Make the unused values NaNs
# NaN values are automatically dropped below with stack()
# zip creates a list of tuples from a tuple of arrays
for coord in zip(*tril_index):
    corr_values.iloc[coord[0], coord[1]] = np.NaN
    
# Stack the data and convert to a data frame
corr_values = (corr_values
               .stack() # multi-index stacking of a matrix: [m1:(m11, m12,...), m2:(m21, m22,...), ...]
               .to_frame() # convert in dataframe
               .reset_index() # new index
               .rename(columns={'level_0':'feature1', # new column names
                                'level_1':'feature2',
                                0:'correlation'}))

# Get the absolute values for sorting
corr_values['abs_correlation'] = corr_values.correlation.abs()

# We have many correlation values, because we have 550+ features!
corr_values.shape # (157080, 4)

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Histogram of correlations
sns.set_context('talk')
sns.set_style('white')
ax = corr_values.abs_correlation.hist(bins=50, figsize=(12, 8))
ax.set(xlabel='Absolute Correlation', ylabel='Frequency');

# The most highly correlated values
corr_values.sort_values('correlation', ascending=False).query('abs_correlation>0.8')

### -- 3. Split dataset maintaining class ratios: StratifiedShuffleSplit

# Orginal class ratios
data.Activity.value_counts(normalize=True)

# StratifiedShuffleSplit allows to split the dataset
# into the desired numbers of train-test subsets
# while still maintaining the ratio of the predicted classes in the original/complete dataset
from sklearn.model_selection import StratifiedShuffleSplit

# Instantiate the StratifiedShuffleSplit object with its parameters
strat_shuf_split = StratifiedShuffleSplit(n_splits=1, # 1 split: 1 train-test
                                          test_size=0.3, 
                                          random_state=42)

# Get the split indexes
train_idx, test_idx = next(strat_shuf_split.split(X=data[feature_cols], y=data.Activity))

# Create the dataframes using the obtained split indices
X_train = data.loc[train_idx, feature_cols]
y_train = data.loc[train_idx, 'Activity']

X_test  = data.loc[test_idx, feature_cols]
y_test  = data.loc[test_idx, 'Activity']

# Always check that the ratios are OK
y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)

### -- 4. Define and fit logistic regression models with and without regularization

# Logistic regression is originally a binary classification tool
# but the sklearn implementation handles multiple classes.
# Depending on the solver, different approaches are taken.
# The liblinear solver uses the one-vs-rest approach; so the model is fit N times,
# being N the number of classses we have.
# liblinear works nice for smaller datasets, read the documentation for more.
# Note: we can add regularization to the LogisticRegression objects in the parameters:
# penalty: l1, l2
# C: 1/lambda -> the larger the less regularization strength
# Also, we can use the cross-validation version to detect the optimum C value
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

# Standard logistic regression
lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)

# L1 regularized logistic regression: it takes a while
lr_l1 = LogisticRegressionCV(Cs=10, cv=4, penalty='l1', solver='liblinear').fit(X_train, y_train)

# L2 regularized logistic regression
lr_l2 = LogisticRegressionCV(Cs=10, cv=4, penalty='l2', solver='liblinear').fit(X_train, y_train)

### -- 5. Compare the magnitude of the coefficients of each model in a multi-level dataframe

# Since we have a multi-class classification model with 6 classes and 561 features
# we get 6 x 561 coefficients.
# Each class-feature coefficient is the strength of the effect that feature
# has on the log odds ratio of the class
lr.coef_.shape

# Combine all the coefficients into a dataframe
coefficients = list()

coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [lr, lr_l1, lr_l2]

for lab,mod in zip(coeff_labels, coeff_models):
    coeffs = mod.coef_ # 6 x 561
    # Create multi-index columns:
    #          lr          l1          l2
    # 0 1 2 3 4 5 0 1 2 3 4 5 0 1 2 3 4 5 
    coeff_label = pd.MultiIndex(levels=[[lab], [0,1,2,3,4,5]], 
                                codes=[[0,0,0,0,0,0], [0,1,2,3,4,5]])
    coefficients.append(pd.DataFrame(coeffs.T, columns=coeff_label))

# Create dataframe from list of dataframes
coefficients = pd.concat(coefficients, axis=1)

# Show 10 random coefficient values
coefficients.sample(10)

# All coefficients of all the models:
# 561 features x (3 models * 6 classes)
coefficients.shape

# Prepare six separate plots for each of the multi-class coefficients.
# All feature coefficients for each class are plotted
# differentiating the 3 models.
fig, axList = plt.subplots(nrows=3, ncols=2)
axList = axList.flatten()
fig.set_size_inches(10,10)

for ax in enumerate(axList):
    loc = ax[0]
    ax = ax[1]
    
    data = coefficients.xs(loc, level=1, axis=1)
    data.plot(marker='o', ls='', ms=2.0, ax=ax, legend=False)
    
    if ax is axList[0]:
        ax.legend(loc=4)
        
    ax.set(title='Coefficient Set '+str(loc))

plt.tight_layout()

### -- 6. Predict the probabilities and the classes of the test split for each model

# Predict the class and the probability for each model
# using the test split.
# The class and the probability are important: predict() and predict_proba()
y_pred = list()
y_prob = list()

coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [lr, lr_l1, lr_l2]

for lab,mod in zip(coeff_labels, coeff_models):
    y_pred.append(pd.Series(mod.predict(X_test), name=lab))
    y_prob.append(pd.Series(mod.predict_proba(X_test).max(axis=1), name=lab))
    
y_pred = pd.concat(y_pred, axis=1)
y_prob = pd.concat(y_prob, axis=1)

y_pred.head()
# lr  l1  l2
# 0 3 3 3
# 1 5 5 5
# ...
y_prob.head()
# lr  l1  l2
# 0 0.998939  0.998965  0.999757
# 1 0.988165  0.999485  0.999998
# ...

# Which are the data-points that obtained a different class
# prediction dpeending on the model?
y_pred[y_pred.lr != y_pred.l1]

### -- 7. Compute the metrics for each model: precision, recall, f1, accuracy, roc-auc, confusion matrix

# For each model, we compute the most important metrics
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

metrics = list() # precision, recall, f1, accuracy, roc-auc
cm = dict() # confusion matrix

# We need to pass the results of each model separately
for lab in coeff_labels:

    # Precision, recall, f-score from the multi-class support function
    # Since we have multiple classes, there is one value for each class for
    # precision, recall, F1 and support.
    # The support is the number of occurrences of each class in ``y_true``.
    # However, we can compute the weighted average to get a global value with average='weighted'.
    # Then, support doesn't make sense.
    # Without the average parameter, we get arrays of six values for each metric,
    # one item in each array for each class.
    precision, recall, fscore, _ = score(y_test, y_pred[lab], average='weighted')
    
    # The usual way to calculate accuracy
    # Accuracy is for the complete dataset (ie., all classes).
    accuracy = accuracy_score(y_test, y_pred[lab])
    
    # ROC-AUC scores can be calculated by binarizing the data
    # label_binarize performs a one-hot encoding,
    # so from an integer class we get an array of one 1 and the rest 0s.
    # This is necessary for computing the ROC curve, since the target needs to be binary!
    # Again, to get a single ROC-AUC from the 6 classes, we pass average='weighted'
    auc = roc_auc_score(label_binarize(y_test, classes=[0,1,2,3,4,5]),
              label_binarize(y_pred[lab], classes=[0,1,2,3,4,5]), 
              average='weighted')
    
    # Last, the confusion matrix
    cm[lab] = confusion_matrix(y_test, y_pred[lab])
    
    metrics.append(pd.Series({'precision':precision, 'recall':recall, 
                              'fscore':fscore, 'accuracy':accuracy,
                              'auc':auc}, 
                             name=lab))

metrics = pd.concat(metrics, axis=1)

metrics
# lr  l1  l2
# precision 0.984144  0.983514  0.984148
# recall  0.984142  0.983495  0.984142
# fscore  0.984143  0.983492  0.984143
# accuracy  0.984142  0.983495  0.984142
# auc 0.990384  0.989949  0.990352

# Confusion matrix plots: one for each model
# Actual vs. Predicted
fig, axList = plt.subplots(nrows=2, ncols=2)
axList = axList.flatten()
fig.set_size_inches(12, 10)

axList[-1].axis('off') # we have 2x2 subplots, but want to show only the first 3 

for ax,lab in zip(axList[:-1], coeff_labels):
    sns.heatmap(cm[lab], ax=ax, annot=True, fmt='d');
    ax.set(title=lab);
    
plt.tight_layout()

# Always check the classes which are more often confused.
# Why are they similar? How could we differentiate them? Do we need more data?
# We see that the most confused classes are 1 & 2
# We can check their meaning with the label encoder
# 'SITTING', 'STANDING'
le.classes_

```

### 1.10 Python Example: Food Items

This notebook shows a classification example in which 13260 food items with 17 nutrient values (features) are labelled (target) as to be consumed:

- in moderation,
- less often,
- more often.

The dataset consists of cleaned numerical values which only need to be scaled. There are few new things compared to the previous notebook. In the following, selected code parts are summarized, with the most important new concepts:

1. Stratified data splits can be performed directly with `train_test_split`
2. Definition of multinomial Logistic Regression with its solver
3. Practical classification metric computation for multinomial cases
4. Coefficient importance for multinomial classification

```python

### -- 1. Stratified data splits can be performed directly with `train_test_split`

# First, let's split the training and testing dataset
# Another way of mainatining the class ratios in the split is using stratify
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = rs)

### -- 2. Definition of multinomial Logistic Regression with its solver

# L1 penalty to shrink coefficients without removing any features from the model
penalty= 'l1'
# We choose our classification problem to define it multinomial in contrast to using
# the one-vs-rest method. In practice, nothing changes for the user, but we need to choose
# other solvers.
multi_class = 'multinomial'
# Use saga for L1 penalty and multinomial classes
solver = 'saga'
# Max iteration = 1000
max_iter = 1000

# L2 penalty to shrink coefficients without removing any features from the model
penalty= 'l2'
# Our classification problem is multinomial
multi_class = 'multinomial'
# Use lbfgs for L2 penalty and multinomial classes
solver = 'lbfgs'
# Max iteration = 1000
max_iter = 1000

# Define a logistic regression model with above arguments
l1_model = LogisticRegression(random_state=rs, penalty=penalty, multi_class=multi_class, solver=solver, max_iter = 1000)
# Define a logistic regression model with above arguments
l2_model = LogisticRegression(random_state=rs, penalty=penalty, multi_class=multi_class, solver=solver, max_iter=max_iter)

### -- 3. Practical classification metric computation for multinomial cases

def evaluate_metrics(yt, yp):
    results_pos = {}
    results_pos['accuracy'] = accuracy_score(yt, yp)
    precision, recall, f_beta, _ = precision_recall_fscore_support(yt, yp)
    results_pos['recall'] = recall
    results_pos['precision'] = precision
    results_pos['f1score'] = f_beta
    return results_pos
evaluate_metrics(y_test, l2_preds)
# {'accuracy': 0.774132730015083,
# 'recall': array([0.87218045, 0.73220641, 0.35353535]),
# 'precision': array([0.73001888, 0.8346856 , 0.90909091]),
# 'f1score': array([0.79479274, 0.78009479, 0.50909091])}

### -- 4. Coefficient importance for multinomial classification

# Even tough we have a multinomial distribution under the hood
# we still get a set of feature coefficients for each class
l1_model.coef_

# Extract and sort feature coefficients
# We have one feature coefficient set for each class
# We pass the class number label_index to extract the values of that class
# in a sorted way.
# We take only coefficient values that are larger than 0.01 in margnitude
def get_feature_coefs(regression_model, label_index, columns):
    coef_dict = {}
    for coef, feat in zip(regression_model.coef_[label_index, :], columns):
        if abs(coef) >= 0.01:
            coef_dict[feat] = coef
    # Sort coefficients and create dictionary of them
    # coefficient_name: coefficient_value
    coef_dict = {k: v for k, v in sorted(coef_dict.items(), key=lambda item: item[1])}
    return coef_dict

# Generate bar colors based on if value is negative or positive
def get_bar_colors(values):
    color_vals = []
    for val in values:
        if val <= 0:
            color_vals.append('r')
        else:
            color_vals.append('g')
    return color_vals

# Visualize coefficients
def visualize_coefs(coef_dict):
    features = list(coef_dict.keys())
    values = list(coef_dict.values())
    y_pos = np.arange(len(features))
    color_vals = get_bar_colors(values)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.barh(y_pos, values, align='center', color=color_vals)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    # labels read top-to-bottom
    ax.invert_yaxis()  
    ax.set_xlabel('Feature Coefficients')
    ax.set_title('')
    plt.show()

# Get the coefficents for Class 1, Less Often
coef_dict = get_feature_coefs(l1_model, 1, feature_cols)
visualize_coefs(coef_dict)

```
