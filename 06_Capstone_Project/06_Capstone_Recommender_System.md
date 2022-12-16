# Deep Learning and Reinforcement Learning

These are my notes and the code of the [IBM Machine Learning Professional Certificate](https://www.coursera.org/professional-certificates/ibm-machine-learning) offered by IBM & Coursera.

The Specialization is divided in 6 courses, and each of them has its own folder with its guide & notebooks:

1. [Exploratory Data Analysis for Machine Learning](https://www.coursera.org/learn/ibm-exploratory-data-analysis-for-machine-learning?specialization=ibm-machine-learning)
2. [Supervised Machine Learning: Regression](https://www.coursera.org/learn/supervised-machine-learning-regression?specialization=ibm-machine-learning)
3. [Supervised Machine Learning: Classification](https://www.coursera.org/learn/supervised-machine-learning-classification?specialization=ibm-machine-learning)
4. [Unsupervised Machine Learning](https://www.coursera.org/learn/ibm-unsupervised-machine-learning?specialization=ibm-machine-learning)
5. [Deep Learning and Reinforcement Learning](https://www.coursera.org/learn/deep-learning-reinforcement-learning?specialization=ibm-machine-learning)
6. [Machine Learning Capstone: Deployment of a Recommender System](https://www.coursera.org/learn/machine-learning-capstone?specialization=ibm-machine-learning)

This file focuses on the **sixth course: Machine Learning Capstone: Deployment of a Recommender System**.

The goal of this course is to build a project in which a recommender system is implemented trying different machine learning techniques.

Mikel Sagardia, 2022.  
No guarantees

## Table of Contents

- [Deep Learning and Reinforcement Learning](#deep-learning-and-reinforcement-learning)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
    - [1.1 Introduction to Recommender Systems](#11-introduction-to-recommender-systems)
    - [1.2 Project Description](#12-project-description)
  - [2. Exploratory Data Analysis on Online Course Enrollment Data](#2-exploratory-data-analysis-on-online-course-enrollment-data)
    - [Download and Analyze Dataset](#download-and-analyze-dataset)
  - [3. Unsupervised-Learning Based Recommender System](#3-unsupervised-learning-based-recommender-system)
  - [4. Supervised-Learning Based Recommender System](#4-supervised-learning-based-recommender-system)
  - [5. Deployment and Presentation](#5-deployment-and-presentation)
  - [6. Project Submission](#6-project-submission)

## 1. Introduction

The notebooks and the code associated to this module and the project are located in [`./lab`](https://github.com/mxagar/machine_learning_ibm/tree/main/06_Capstone_Project/lab).

### 1.1 Introduction to Recommender Systems

:warning: For a more theoretical introduction, check my notes in [`ML_Anomaly_Recommender.md`](https://github.com/mxagar/machine_learning_coursera/blob/main/07_Anomaly_Recommender/ML_Anomaly_Recommender.md).

Even though people's taste might vary, they follow patterns: they like things of the same category, with similar contents, etc. Recommender systems are everywhere and they suggest us many things based on a model:

- Books to buy.
- Where to eat.
- Movies to see.
- Jobs to apply to.
- Who to be friends with.
- News to read.
- etc.

One could argue that recommender systems are good for the two parties involved in the transaction:

- The service provider, because they sell more.
- The consumer, because they get more of what they like.

There are two main types of recommender systems:

1. Content-based: "Show me more of the same of what I've liked before".
   - The system figures out the elements the user likes and tries to find items that share those aspects.
2. Collaborative Filtering: "Tell me what's popular among my neighbors, I also might like it".
   - Users are put into groups of similarity, and the items popular in those groups are suggested to the peers that haven't experienced them.
3. Hybrid: a combination of both.

Implementation can be:

1. Memory-based
   - Entire user-item dataset used.
   - Items and users are represented as vectors and their similarities can be compute: cosine, correlation, Euclidean distance, etc.
2. Model-based
   - A model of users is developed to learn their preferences.
   - Models can be anything: regression, classification, clustering, etc.

### 1.2 Project Description

We need to build a recommender system which suggests AI courses to students. The system will be deployed as a Streamlit web app.

The following picture shows the different components of the project:

![Project Workflow](./pics/project_workflow.png)

Tasks:

- Collecting and understanding data
- EDA
- Extracting Bag of Words (BoW) features from course textual content
- Calculating course similarity using BoW features
- Building content-based recommender systems using various unsupervised learning algorithms, such as: Distance/Similarity measurements, K-means, Principal Component Analysis (PCA), etc.
- Building collaborative-filtering recommender systems using various supervised learning algorithms: K Nearest Neighbors, Non-negative Matrix Factorization (NMF), Neural Networks, Linear Regression, Logistic Regression, RandomForest, etc.
- Creating an insightful and informative slideshow and presenting it to your peers

## 2. Exploratory Data Analysis on Online Course Enrollment Data

### Download and Analyze Dataset

Notebook: [`lab_jupyter_eda.ipynb`](https://github.com/mxagar/machine_learning_ibm/blob/main/06_Capstone_Project/lab/)

Nothing really new is done in the notebook/exercise.

The dataset consists on two files:

1. `course_genre.csv`: `(307, 16)`: course id, title and binary values of topics covered in each course.
2. `ratings.csv`: `(233306, 3)`: user id, course id and rating of each course by the user; the rating has only two possible values: `2: enrolled, not finished`, `3: enrolled and finished`.

Steps followed:

- All titles are joined to created a `wordcloud`.
- Course counts for topics are analyzed: sorted according to counts (popularity of each topic).
- Users with most enrollments are ranked.
- Courses with most enrollments are ranked: 20 most popular.
- A join (`merge()`) is performed to get course names.

## 3. Unsupervised-Learning Based Recommender System

## 4. Supervised-Learning Based Recommender System

## 5. Deployment and Presentation

## 6. Project Submission