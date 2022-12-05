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
    - [1.9 Loss Functions](#19-loss-functions)
  - [2. Keras: Basics](#2-keras-basics)
  - [3. Convolutional Neural Networks (CNNs)](#3-convolutional-neural-networks-cnns)
    - [3.1 Main Concepts](#31-main-concepts)
    - [3.2 Lab: CNNs on CIFAR-10](#32-lab-cnns-on-cifar-10)
    - [3.3 Transfer Learning](#33-transfer-learning)
    - [3.4 Lab: Transfer Learning](#34-lab-transfer-learning)
    - [3.5 Popular CNN Architectures](#35-popular-cnn-architectures)

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

### 1.9 Loss Functions

See also: [keras/losses](https://www.tensorflow.org/api_docs/python/tf/keras/losses).

- Regression:
    - `MeanSquaredError()`
    - `MeanAbsoluteError()`
    - `cosine_similarity()`
    - ...
- Classification:
    - `BinaryCrossentropy()`
    - Multi-class: `CategoricalCrossentropy()`
    - ...

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

## 3. Convolutional Neural Networks (CNNs)

### 3.1 Main Concepts

Very brief notes on the main concepts:

- Fully connected too many parameters
- Images contain object that need to be identified with invariance wrt
  - Translation
  - Size/scale
  - Rotation
- Pixel values change few values in their neighborhood unless we have edges or salient texture.
- Convolutional layers: 
  - Kernels or grid are overlaid on image patches centered in a pixel.
  - Kernel grid is multiplied one-by-one to pixel values beneath.
  - Sum is the result of the center pixel.
  - Kernel is strided/swept along the X & Y axes of the image.
- Kernels are filters: low pass (blur) or high pass (edges).
- The weights of the kernels are learnt.
- Convolutional kernels require much less parameters than dense/fully connected layers and are much better suited to the spatial information contained in images.

Common settings/parameters:

- **Kernel size**: width and height pixels of the filters; usually square kernels are applied with odd numbers: `3 x 3` (recommended), `5 x 5` (less recommended, because more parameters).
- **Padding**: so that we can use corner/edge pixels as centers for the kernels, we add extra pixels on the edges corner; usually, the added pixels have value 0, i.e., *zero-padding*.
  - If we add no padding, the output activation map will be smaller than the input.
  - To conserve image size: `padding = (F-1)/2` with `F` kernel/filter size.
- **Stride**: movement of the kernel in X & Y directions.
  - Usually same stride is used in X & Y.
  - If `stride > 2` we're dividing the image size by `stride`.
- **Depth**: number of channels; we have input and output channels.
  - Each input image has `n` channels.
  - Each output image/map has `N` channels.
  - We have `N` filters, each with `n` kernels applied to the input image.

Another important layer in CNNs: **Pooling**: Pooling reduces image size by mapping an image patch to a value. Commonly `2 x 2` pooling is done, using as stride the pooling window size (i.e., no overlap). We have different types:

- Max-pooling.
- Average-pooling.

### 3.2 Lab: CNNs on CIFAR-10

In this section, this notebook is explained:

`05e_DEMO_CNN.ipynb`

In it, the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset is used, which consists of 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. Check the current [performance results here](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html).

The 10 classes are:

<ol start="0">
<li> airplane
<li> automobile
<li> bird
<li> cat
<li> deer
<li> dog
<li> frog
<li> horse
<li> ship
<li> truck
</ol>

In the notebook, the following steps are carried out:

1. Imports
2. Load dataset
3. Prepare dataset: encode & scale
4. Define model
5. Train model
6. Evaluate model

```python

###
# 1. Imports
##

import keras
#from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
#from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

###
# 2. Load dataset
##

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape) # (50000, 32, 32, 3)
print(x_train.shape[0], 'train samples') # 50000 train samples
print(x_test.shape[0], 'test samples') # 10000 test samples

###
# 3. Prepare dataset: encode & scale
##

# Each image is a 32 x 32 x 3 numpy array
x_train[444].shape

# Visualize the images
print(y_train[444]) # [9]
plt.imshow(x_train[444]);

# One-hot encoding in Keras/TF
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

y_train[444] # [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]

# Let's make everything float and scale
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

###
# 4. Define model
##

# Let's build a CNN using Keras' Sequential capabilities

model = Sequential()

# Conv2D has these parameters:
# - filters: the number of filters used (= depth of output)
# - kernel_size: an (x,y) tuple giving the height and width of the kernel
# - strides: an (x,y) tuple giving the stride in each dimension; default and common (1,1)
# - input_shape: required only for the first layer (= image channels)
# - padding: "valid" = no padding, or "same" = zeros evenly;
# When padding="same" and strides=1, the output has the same size as the input
# Otherwise, general formula for the size:
# W_out = (W_in + 2P - F)/S + 1; P: "same" = (F-1)/2 ?
model.add(Conv2D(filters=32,
                   kernel_size=(3,3), # common
                   padding='same', # 
                   strides=(1,1), # common, default value
                   input_shape=x_train.shape[1:]))
# We can specify the activation as a layer (as done here)
# or in the previous layer as a parameter
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
# Parameters od MaxPooling2D:
# - pool_size: the (x,y) size of the grid to be pooled; 2x2 (usual) halvens the size
# - strides: assumed to be the pool_size unless otherwise specified
# - padding: assumed "valid" = no padding
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten appears when going from convolutional layers to
# fully connected layers.
model.add(Flatten())
model.add(Dense(units=512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(units=num_classes))
model.add(Activation('softmax'))

# Always check number of paramaters!
model.summary()

###
# 5. Train model
##

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(lr=0.0005)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=5,
              validation_data=(x_test, y_test),
              shuffle=True)

###
# 6. Evaluate model
##

# Validation loss and Validation accuracy
model.evaluate(x_test, y_test)

# Manual computation of the accuracy
import numpy as np
from sklearn.metrics import accuracy_score

y_pred = model.predict_classes(x_test)
y_true = np.argmax(y_test, axis=1) # undo one-hot encoding
print(accuracy_score(y_true, y_pred))

```

### 3.3 Transfer Learning

- First layers learn filter weights able to detect edges; as we go deeper, the learned shapes start getting more complex.
- Edges and simple shapes generalize well, that's why we can apply transfer learning: we take the backbone/initial part of the trained complex CNN and re-use it for our application. Last layers (the classifier) are the ones that are trained.
- Possible transfer learning techniques:
    - Only the classifier: transfer learning
    - Additional training of the pre-trained network: fine tuning
        - We can choose to re-train the entire network using as initialization the pre-trained weights or we can select an amount of layers.
    - Which one should we use?
        - The more similar the datasets, the less fine-tuning necessary.
        - The more data we have available, the more the model will benefit from fine-tuning

### 3.4 Lab: Transfer Learning

In this section, this notebook is explained:

`05f_DEMO_Transfer_Learning.ipynb`

In it, the MNIST dataset is used. First, a model is trained with the digits `0-4`; then, we freeze the *feature layer* weights and apply transfer learning to the model in which only the *classifier layers* are re-trained with the digits `5-9`. The training is faster because we train only the classifier.

Most important steps:

1. Imports
2. Define parameters
3. Define data pre-processing + training function
4. Load dataset + split
5. Train: Digits 5-9
6. Freeze feature layers and re-train with digits 0-4

```python

###
# 1. Imports
###

import datetime
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
#from tensorflow import keras
#from tensorflow.keras.datasets import mnist
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
#from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from tensorflow.keras import backend as K

# Used to help some of the timing functions
now = datetime.datetime.now

###
# 2. Define parameters
###

# set some parameters
batch_size = 128
num_classes = 5
epochs = 5

# set some more parameters
img_rows, img_cols = 28, 28
filters = 32
pool_size = 2
kernel_size = 3

## This just handles some variability in how the input data is loaded
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

###
# 3. Define data pre-processing + training function
###

# To simplify things, write a function to include all the training steps
# As input, function takes a model, training set, test set, and the number of classes
# Inside the model object will be the state about which layers we are freezing and which we are training
def train_model(model, train, test, num_classes):
    # train = (x_train, y_train)
    # test = (x_test, y_test)
    x_train = train[0].reshape((train[0].shape[0],) + input_shape) # (60000, 28, 28, 1)
    x_test = test[0].reshape((test[0].shape[0],) + input_shape) # (60000, 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    # Measure time
    t = now()
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    print('Training time: %s' % (now() - t)) # Measure time

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

###
# 4. Load dataset + split
###

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# create two datasets: one with digits below 5 and the other with 5 and above
x_train_lt5 = x_train[y_train < 5]
y_train_lt5 = y_train[y_train < 5]
x_test_lt5 = x_test[y_test < 5]
y_test_lt5 = y_test[y_test < 5]

x_train_gte5 = x_train[y_train >= 5]
y_train_gte5 = y_train[y_train >= 5] - 5
x_test_gte5 = x_test[y_test >= 5]
y_test_gte5 = y_test[y_test >= 5] - 5

x_train.shape # (60000, 28, 28)
y_train.shape # (60000,)
input_shape # (28, 28, 1)

###
# 5. Define model: feature layers + classifier
###

# Define the "feature" layers.  These are the early layers that we expect will "transfer"
# to a new problem.  We will freeze these layers during the fine-tuning process
feature_layers = [
    Conv2D(filters, kernel_size,
           padding='valid',
           input_shape=input_shape),
    Activation('relu'),
    Conv2D(filters, kernel_size),
    Activation('relu'),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.25),
    Flatten(),
]

# Define the "classification" layers.  These are the later layers that predict the specific classes from the features
# learned by the feature layers.  This is the part of the model that needs to be re-trained for a new problem
classification_layers = [
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_classes),
    Activation('softmax')
]

# We create our model by combining the two sets of layers as follows
model = Sequential(feature_layers + classification_layers)

# Let's take a look: see "trainable" parameters
model.summary()

###
# 5. Train: Digits 5-9
###

# Now, let's train our model on the digits 5,6,7,8,9
train_model(model,
            (x_train_gte5, y_train_gte5),
            (x_test_gte5, y_test_gte5), num_classes)


###
# 6. Freeze feature layers and re-train with digits 0-4
###

# Freeze only the feature layers
for l in feature_layers:
    l.trainable = False

model.summary() # We see that the "trainable" parameters are less

train_model(model,
            (x_train_lt5, y_train_lt5),
            (x_test_lt5, y_test_lt5), num_classes)

```

### 3.5 Popular CNN Architectures

- LeNet
  - Yann LeCun, 1990
  - First CNN
  - Back & white images; tested with MNIST
  - Three times: Conv 5x5 + Subsampling (pooling); then, 2 fully connected layers. 
- AlexNet
  - It popularized the CNNs.
  - Turning point for modern Deep Learning.
  - 16M parameters.
  - They parallelized the network to train in 2 GPUs.
  - Data augmentation was performed to prevent overfitting and allow generalization.
  - ReLUs were used: huge step at the time.
- VGG
  - It simplified the choice of sizes: it uses only 3x3 kernels and deep networks, which effectively replace larger convolutions.
  - The receptive field of two 3x3 kernels is like the receptive field of one 5x5 kernel, but with less parameters! As we go larger, the effect is bigger.
  - VGG showed that many small kernels are better: deep networks.
- Inception
  - The idea is that we often don't know which kernel size should be better applied; thus, instead of fixing on size, we apply several in parallel for each layer and then we concatenate the results.
  - In order to control the depth for each branch, 1x1 convolutions were introduced.
  - See the architecture schematic below.
- ResNet
  - VGG showed the power of deep networks; however, from a point on, as we goo deeper, the performance decays because:
    - Early layers are harder too update
    - Vanishing gradient
  - ResNet proposed learning the residuals; in practice, that means that we add the output from the 2nd previous layer to the current output. As a result, the delta is learned and the previous signal remains untouched. With that, we alleviate considerably the vanishing gradient issue and the networks can be **very deep**!

![Inception: Block](./pics/inception_1.jpg)

![Inception: Architecture](./pics/inception_2.jpg)