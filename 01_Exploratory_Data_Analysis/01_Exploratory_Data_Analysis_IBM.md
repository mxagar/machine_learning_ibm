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

### 2.1 Retrieving Data


