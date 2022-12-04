# Deep Learning and Reinforcement Learning

These are my notes and the code of the [IBM Machine Learning Professional Certificate](https://www.coursera.org/professional-certificates/ibm-machine-learning) offered by IBM & Coursera.

The Specialization is divided in 6 courses, and each of them has its own folder with its guide & notebooks:

1. [Exploratory Data Analysis for Machine Learning](https://www.coursera.org/learn/ibm-exploratory-data-analysis-for-machine-learning?specialization=ibm-machine-learning)
2. [Supervised Machine Learning: Regression](https://www.coursera.org/learn/supervised-machine-learning-regression?specialization=ibm-machine-learning)
3. [Supervised Machine Learning: Classification](https://www.coursera.org/learn/supervised-machine-learning-classification?specialization=ibm-machine-learning)
4. [Unsupervised Machine Learning](https://www.coursera.org/learn/ibm-unsupervised-machine-learning?specialization=ibm-machine-learning)
5. [Deep Learning and Reinforcement Learning](https://www.coursera.org/learn/deep-learning-reinforcement-learning?specialization=ibm-machine-learning)
6. [Specialized Models: Time Series and Survival Analysis](https://www.coursera.org/learn/time-series-survival-analysis?specialization=ibm-machine-learning)

This file focuses on the **fifth course: Deep Learning and Reinforcement Learning**

In contrast to the other courses of the Specialization, I've taken few notes this time; I you're interested in more details on Deep Learning, I suggest you to visit my repository [deep_learning_udacity](https://github.com/mxagar/deep_learning_udacity).

Also, **check my [`DL_Keras_Guide.md`](https://github.com/mxagar/deep_learning_udacity/blob/main/02_Keras_Guide/DL_Keras_Guide.md)**, located in that repository.

Mikel Sagardia, 2022.  
No guarantees

## Table of Contents

- [Deep Learning and Reinforcement Learning](#deep-learning-and-reinforcement-learning)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
    - [1.1 Introduction to Neural Networks](#11-introduction-to-neural-networks)
    - [1.2 Gradient Descend](#12-gradient-descend)
    - [1.3 Lab: Gradient Descend and Neural Networks](#13-lab-gradient-descend-and-neural-networks)
    - [1.4 Backpropagation and Activation Functions](#14-backpropagation-and-activation-functions)
    - [1.5 Lab: Backpropagation](#15-lab-backpropagation)
    - [1.6 Regularization](#16-regularization)
    - [1.7 Optimizers](#17-optimizers)
    - [1.8 Data Shuffling](#18-data-shuffling)
  - [2. Keras: Basics](#2-keras-basics)

## 1. Introduction

### 1.1 Introduction to Neural Networks

Concepts introduced:

- Nodes = neurons.
- Input: `x` (vector).
- Weights, bias: packed into matrices: `W (n x m), n input nodes & m output nodes`.
- Raw output of a neuron: linear combination of inputs scaled by weights: `z` (vector).
- Output of a neuron, `z`, is activated, resulting in the input for next layer: `a = f(z)` (vector).
- Activation function for non-linearity: sigmoid: `f = sigmoid()`.
- Perceptron model = logistic regression!
- Nice property of sigmoid: `f' = f(1-f)`.
- By stacking neurons in layers and layers in networks, we can have very complex decision boundaries.
- MLP = Multi-Layer Perceptron.
- There are MLP models in Scikit-Learn, but we'll use Tensorflow/Keras.
- If multi-class classification, last activation is softmax.
- Always scale data so that weight values remain in the same range!
- Deep learning requires large datasets (i.e., many rows).

Vector sizes: even though I wrote that `x`, `z`, `a` are vectors, they can indeed be matrices. Indeed, that's what happens with batches:

    x (b, n)
    W (n, m)
    z = x.W (b, n).(n, m) = (b, m)
    a = f(z) (b, m)

### 1.2 Gradient Descend

Concepts introduced:

- Error: difference between prediction and real label/value.
- Cost function = error: expressed in function of model parameters: `J()`.
- Minimum cost function is found in gradient descend.
- The gradient of the cost function points in the direction of the **largest increase**; so **we take the opposite direction**.
- Gradient: vector of partial derivatives of `J` wrt. each parameter: `dJ/dw_i`.
- Weight update with learning rate: `w_new <- w_old - learning_rate*[dJ/dw_i]`.
- Stochastic gradient descend: in gradient descend we compute the error/cost considering all samples; in stochastic gradient descend we take one random sample. That's more noisy.
- Mini-batch gradient descend: we compute the cost with a batch of samples. Best of both worlds: less noisy and less intensive.
- Epoch: single pass through all training data
  - In batch gradient descend an epoch is one step.
  - In SGD on epoch is many steps, as many as samples.
  - In mini-batch GD, one epoch is also many steps, as many as number of batches.

![Batching Approaches](./pics/batching.jpg)

### 1.3 Lab: Gradient Descend and Neural Networks

Notebooks provided and commented:

- `05a_DEMO_Gradient_Descent.ipynb`
- `05b_LAB_Intro_NN.ipynb`

In the first notebook (`05a`), linear regression is solved  with the closed form formula, gradient descend and stochastic gradient descend. The results are compared. We use gradient descend or its variants for MLPs and deep learning because no closed form formula exists for solving them.

The second notebook (`05b`) has two parts:

- First, a logic gate consisting of a single neuron with 2 input values and a sigmoid activation is built. It has two weights and a bias term. The idea is that we can model many logic gates with a single neuron: `AND, OR, NAND`, etc. However, it fails with `XOR`. The `XOR` logic gate return true only if one input is true and the other false; in order to build such a gate we need to stack two layers of neurons!
- Second, a feed forward pass in a simple neural network is done with matrices.

### 1.4 Backpropagation and Activation Functions

Backpropagation consists in propagating the error derivative backwards in the network so that we can compute every `dJ/dw_i`, i.e., the partial derivative of the cost with respect to each of the weights. Then, with that derivative, we can update each weight. All these derivatives form the gradient.

The derivatives are obtained by applying the chain rule. In practice, we see that the derivatives seem to be the error fed backwards in the network:

![Backpropagation](./pics/backpropagation.png)

![Backpropagation](./pics/backpropagation_2.jpg)

Vanishing gradient problem: since we chain multiplications and the slope of the sigmoid is small at extremes, products become very small, so the gradient vanishes. A solution is to use other activation functions, such as ReLU.

Typical activation functions:

- Sigmoid: problems with vanishing gradient.
- Hyperbolic tangent: stretched sigmoid; also prone to vanishing gradient.
- ReLU: Rectified Linear Unit. Better than sigmoid and `tanh`; bit for negative values we don't learn anything.
- Leaky ReLU: a slope is added for negative values in the ReLU function, so we can learn something. However, leaky ReLUs are not always better than ReLU!

### 1.5 Lab: Backpropagation

Notebook provided and commented: `05c_DEMO_Backpropagation.ipynb`.

A simple neural network with a hidden layer is defined with numpy and the feedforward and backpropagation passes are carried out manually. 

### 1.6 Regularization

Deep NN: Those with >=2 hidden layers; the more hidden layers, the more complex patterns that can be learned. But we risk overfitting = learning noise and not being able to generalize.

Regularization: any technique done to reduce generalization error, but not the training error:

- Regularization penalty in cost function; we can do that also with neural networks.
- Dropout. Typical method in NNs. We need to scale weights with `p` during test time if they were cancelled with `p` probability during training!
- Early stopping: Check validation error and stop as it starts increasing.
- Stochastic GD or mini-batch GD regularize the training, too, because we don't fit the dataset perfectly.

### 1.7 Optimizers

Optimizers perform the weight update; the easiest optimizer is gradient descend:

    w_new <- w_old - learning_rate * [dJ/d_w_i]

However, there are many more optimizers.

**Momentum**: use running average of the previous steps; momentum is the factor that scales the influence of all previous steps. Common value: `eta = 0.9`. Often times, the learning rate is chosen as `alpha = 1 - eta`. The effect of using momentum is that we smooth out the steps, as compared to stochastic/gradient descend.

![Momentum](./pics/momentum.jpg)

**Nesterov Momentum**: momentum alone can overshoot the optimum solution. Nesterov momentum controls that overshooting. The effect is that the steps are even more smooth.

![Nesterov](./pics/nesterov.jpg)

**AdaGrad**: Adaptive gradient algorithm:

- Frequently updated weights are updated less.
- We track the value `G`, sum of previous gradients, which increases every iteration and divide each learning rate with it.
- Effect: as we get closer to the solution, the learning rate is smaller, so we avoid overshooting.

![AdaGrad](./pics/adagrad.jpg)

**RMSProp**: Root mean square propagation. Similar to AdaGrad, but more efficient. It tracks `G`, but older gradients have a smaller weight; the effect is that newer gradients have more impact.

**Adam**: Momentum and RMSProp combined. We have two parameters to tune, which have these default values:

- `beta1 = 0.9`
- `beta2 = 0.999`

![Adam](./pics/adam.jpg)

Which one should we use? Adam and RMSProp are very popular and work very well: they're fast. However, if we have convergence issues, we should try simple optimizers, like stochastic gradient descend.

### 1.8 Data Shuffling

We need to shuffle our data every epoch to avoid cyclical movement and doing the same path every epoch! When we shuffle, the batches are different each time, and of course, the order is not the same.

## 2. Keras: Basics

In this section, the following notebook is shown, which introduces how the workflow and calls in Tensorflow/Keras work:

`05d_LAB_Keras_Intro.ipynb`

The [Diabetes Dataset](https://archive.ics.uci.edu/ml/datasets/diabetes) is used and a random forest is compared to a very simple neural network.

Random forests and gradient boosting methods (e.g., XGBoost) often outperform neural networks for tabular data; that's the case also here.

With neural network it's very important to use a validation split and to plot the learning curves. We should also vary the used optimizer, the learning rate, activation functions, etc.

Notes:

- Neural networks are not the best option for tabular data.
- Neural networks require large datasets (many rows).

Most important code blocks:

1. Imports
2. Load and prepare dataset: split + normalize
3. Define model: Sequential + Compile (Optimizer, Loss, Metrics)
4. Train model
5. Evaluate model and Inference
6. Save and Load

```python

#####
## 1. Imports
#####

# Import basic ML libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, accuracy_score

# Import Keras objects for Deep Learning
from tensorflow.keras.models  import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import load_model

#####
## 2. Load and prepare dataset: split + normalize
#####

# Load in the data set 
names = ["times_pregnant",
         "glucose_tolerance_test",
         "blood_pressure",
         "skin_thickness",
         "insulin", 
         "bmi",
         "pedigree_function",
         "age",
         "has_diabetes"]
diabetes_df = pd.read_csv('diabetes.csv', names=names, header=0)

print(diabetes_df.shape) # (768, 9): very small dataset to do deep learning

# Split and scale
X = diabetes_df.iloc[:, :-1].values
y = diabetes_df["has_diabetes"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=11111)

normalizer = StandardScaler()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)

#####
## 3. Define model: Sequential + Compile (Optimizer, Loss, Metrics)
#####

# Define a fully connected model 
# - Input size: 8-dimensional
# - Hidden layers: 2 layers, 12 hidden nodes/each, relu activation
# - Dense layers: we specify number of OUTPUT units; for the first layer we specify the input_shape, too
# - Activation: we can either add as layer add(Activation('sigmoid')) or as parameter of Dense(activation='sigmoid')
# - Without an activation function, the activation is linear, i.e. f(x) = x -> regression
# - Final layer has just one node with a sigmoid activation (standard for binary classification)
model = Sequential()
model.add(Dense(units=12, input_shape=(8,), activation='relu'))
model.add(Dense(units=12, input_shape=(8,), activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Summary, parameters
model.summary()

# Compile: Set the the model with Optimizer, Loss Function and Metrics
model.compile(optimizer=SGD(lr = .003),
                loss="binary_crossentropy", 
                metrics=["accuracy"])
# Other options:
# For a multi-class classification problem
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy']) # BUT: balanced
# For a binary classification problem
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy']) # BUT: balanced
# For a mean squared error regression problem
# model.compile(optimizer='adam',
#               loss='mse')
#
# opt = keras.optimizers.Adam(learning_rate=0.01)
# opt = keras.optimizers.SGD(learning_rate=0.01)
# opt = keras.optimizers.RMSprop(learning_rate=0.01)
# ...
# model.compile(..., optimizer=opt)

#####
## 4. Train model
#####

# Train == Fit
# We pass the data to the fit() function,
# including the validation data
# The fit function returns the run history:
# it contains 'val_loss', 'val_accuracy', 'loss', 'accuracy'
run_hist = model.fit(X_train_norm,
                         y_train,
                         validation_data=(X_test_norm, y_test),
                         epochs=200)

#####
## 5. Evaluate model and Inference
#####

# Two kinds of predictions
# One is a hard decision,
# the other is a probabilitistic score.
y_pred_class_nn_1 = model.predict_classes(X_test_norm) # {0, 1}
y_pred_prob_nn_1 = model.predict(X_test_norm) # [0, 1]

# Print model performance and plot the roc curve
print('accuracy is {:.3f}'.format(accuracy_score(y_test,y_pred_class_nn_1))) # 0.755
print('roc-auc is {:.3f}'.format(roc_auc_score(y_test,y_pred_prob_nn_1))) # 0.798

# Plot ROC
def plot_roc(y_test, y_pred, model_name):
    fpr, tpr, thr = roc_curve(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, 'k-')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=.5)  # roc curve for random model
    ax.grid(True)
    ax.set(title='ROC Curve for {} on PIMA diabetes problem'.format(model_name),
           xlim=[-0.01, 1.01], ylim=[-0.01, 1.01])

plot_roc(y_test, y_pred_prob_nn_1, 'NN')

# Learning curves
run_hist_1.history.keys() # ['val_loss', 'val_accuracy', 'loss', 'accuracy']
fig, ax = plt.subplots()
ax.plot(run_hist_1.history["loss"],'r', marker='.', label="Train Loss")
ax.plot(run_hist_1.history["val_loss"],'b', marker='.', label="Validation Loss")
ax.legend()

# Learning curves: Another option
losses = pd.DataFrame(model.history.history)
losses.plot()

# We can further train it!
# That's sensible if curves are descending
run_hist_ = model.fit(X_train_norm, y_train, validation_data=(X_test_norm, y_test), epochs=1000)

# Also: evaluate
# Evaluate the model: Compute the average loss for a new dataset = the test split
model.evaluate(X_test_norm,y_test)

#####
## 6. Save and Load
#####

model.save('my_model.h5')
later_model = load_model('my_model.h5')
later_model.predict(X_test_norm.iloc[101, :])
```