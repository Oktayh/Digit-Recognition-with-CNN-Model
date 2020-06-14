## Libraries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
import time
warnings.filterwarnings("ignore")
start = time.time()
## Dataset importing
# I will use test set for only testing data. Because using unseen data for testing the model,
# It can prevent overfitting. For training the model I will split train set to train and validation.
test = pd.read_csv('dataset/test.csv')
train = pd.read_csv('dataset/train.csv')

## Pre-Processing

# In data set there is 42000 rows and 785 columns
# So we have to check some examples

# first 10 test data
test_head = test.head()
# first 10 train data
train_head = train.head()
# Assigning labels to y_train
y_train = train['label']
# All other features to x_train
x_train = train.drop(labels='label', axis=1)

# Checking null values
print("NULL Value Check for Train X:\n", x_train.isnull().sum())
print("NULL Value Check for Train Y: ", y_train.isnull().sum())
print("NULL Value Check for Testset:\n", test.isnull().sum())
#There is no null values

countplot = sns.countplot(y_train)

## Normalization
x_train = x_train / 255.0
test = test / 255.0
print(x_train.shape)

# Reshaping values to correct format
x_train = x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# convert to one-hot-encoding
from keras.utils.np_utils import to_categorical
# We have numbers 0 to 9. I need to encode that to one hot vector
y_train = to_categorical(y_train, num_classes = 10)

#Shapes of data
print("Shape of X train: ",x_train.shape)
print("Shape of Y train: ",y_train.shape)
print("Shape of ValidationSet:",test.shape)
## Split Train set an Test set
from sklearn.model_selection import train_test_split
#Splitting train set with using sklearn
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=42)
#Shapes of data after splitting
print("X_train shape: {}, X_test: {}".format(X_train.shape,X_val.shape))
print("Y_train shape: {}, Y_test: {}".format(Y_train.shape,Y_val.shape))

import tensorflow as tf
from tensorflow.keras import layers
def my_model():
    # The input shape is the size of the image 28x28 with 1 byte color
    # Creating sequential model
    model = tf.keras.models.Sequential()
    # This is the first convolution
    model.add(tf.keras.layers.Conv2D(16,(3,3),activation=tf.nn.relu,padding = 'Same',input_shape=(28,28,1))),
    # This is the first pooling - 2x2 down sampling filter
    model.add(tf.keras.layers.MaxPooling2D(2,2)),
    # The second convolution
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu,padding = 'Same')),
    # This is the second pooling - 2x2 down sampling filter
    model.add(tf.keras.layers.MaxPooling2D(2, 2)),
    # The third convolution
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu,padding = 'Same')),
    # This is the third pooling - 2x2 down sampling filter
    model.add(tf.keras.layers.MaxPooling2D(2, 2)),
    #Dropout regularizer to avoid overfitting etc.
    model.add(tf.keras.layers.Dropout(0.25)),

    # The fourth convolution
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu,padding = 'Same')),
    # This is the fourth pooling - 2x2 down sampling filter
    model.add(tf.keras.layers.MaxPooling2D(2, 2)),
    # Dropout regularizer
    model.add(tf.keras.layers.Dropout(0.25)),

    # Flatten the results to feed into a DNN
    model.add(tf.keras.layers.Flatten()),
    # 256 neuron hidden layer
    model.add(tf.keras.layers.Dense(256,activation=tf.nn.relu)),
    model.add(tf.keras.layers.Dropout(0.50)),
    # 10 output neurons. It will contain numbers from 0-9.(10 total)
    model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

    #Using the RMSprop optimization algorithm is preferable to stochastic gradient descent (SGD)
    #because RMSprop automates learning-rate tuning for us.
    #-
    # We use a specific form for categorical classifications
    # (>2 classes) called the "categorical_crossentropy".
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model
##asd

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn=my_model)

param_grid = { 'epochs':[10,30],
               'batch_size':[20,100]
               }
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=2)
grid_result = grid.fit(X_train,Y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

##

