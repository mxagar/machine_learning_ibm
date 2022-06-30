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
2. [Linear Regression](#2.-Linear-Regression)

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

