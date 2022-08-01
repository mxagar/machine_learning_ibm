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

We can have higher dimensions for the `x` variable, such that the threshold becomes a linear boundary or a hyperplane.

### 1.3 Classification with Multiple Classes

Muti-class classification can be performed with the *one-vs-all* technique: a binary classification model is composed for each class versus the rest of the classes. This way, each point in feature space gets one probability for each of the classes, and we pick the largest one to create the regions and boundaries.

![Multi-class classification](./pics/multi_class_classification.png)

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

However, we can compute each one of them for each class, i.e., taking the *one-vs-all* approach.

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



### 1.10 Python Example: Food Items

