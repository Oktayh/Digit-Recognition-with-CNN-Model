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
## Plots for images inside of the dataset
nrows,ncols = 10,10
plt.figure(figsize=(10,10))
for digits in range(0,20):
    # There is more than one plot.For ploting Purpose
    plt.subplot(nrows,ncols,digits+1,frameon=False)
    # Next Image of a train set
    next_digit = X_train[digits].reshape(28,28)
    # plt.imshow is used for plotting numerical values
    plt.imshow(next_digit,cmap = "gray")
# Padding for images. Because values overlaps
plt.tight_layout()
plt.show()

## CNN - MODEL

# Build Model
import tensorflow as tf
from tensorflow.keras import layers
def my_model(metrics=['accuracy']):
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
                  metrics=metrics)
    return model

# Train Model

def train_model(model,x,y,epochs,batch_size):

    # Training of the model with train set.
    history = model.fit(x=x,y=y,batch_size=batch_size,validation_data = (X_val,Y_val),epochs=epochs)
    #steps
    epochs  = history.epoch
    hist = pd.DataFrame(history.history)
    return epochs,hist
    print('Train Model defined')

# Plot model curves

def plot_curve(epochs, hist, list_of_metrics):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    #taking merics to plot
    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)

    plt.legend()
    plt.show()


print("Plot Curve Defined.")

## Defining Hyper Parameters.
# Test #1
learning_rate = 0.001
epochs = 30
batch_size = 86
METRICS = [
            #shows accuracy as a result of training phases
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            #shows precision as a result of training phases
            tf.keras.metrics.Precision(name='precision'),
            #shows recall as a result of training phases
            tf.keras.metrics.Recall(name="recall")
            ]

#Assigning function
my_model = my_model(METRICS)

# Train the model on the training set.
epochs, hist = train_model(my_model,X_train, Y_train, epochs, batch_size)

## Curve1 - Accuracy
curves1 = ['accuracy']
plot_curve(epochs, hist, curves1)
## Curve2 - Precision
curves2 = ['precision']
plot_curve(epochs, hist, curves2)
## Curve3 - Recall
curves3 = ['recall']
plot_curve(epochs, hist, curves3)
## Model Summary

from tensorflow.keras.utils import plot_model
plot_model(my_model)

#shows model summary
my_model.summary()



## Model evaluate
my_model.evaluate(X_val, Y_val, verbose=1,batch_size=batch_size)

# Getting results of prediction within a test set
results = my_model.predict(test)
results = np.argmax(results,axis=1)
results = pd.Series(results,name = "Label")

## f1-Score
Mmetric = my_model.history.history
f1_score =  2 * (((Mmetric['precision'][::-1][0] * Mmetric['recall'][::-1][0]))/
    ((Mmetric['precision'][::-1][0] + Mmetric['recall'][::-1][0])))
print('F1 Score of the model: {}'.format(f1_score))


## test data visualization for compare the results (first 20 sample)

nrows,ncols = 10,10
plt.figure(figsize=(10,20))
for digits in range(0,10):
    # There is more than one plot.For ploting Purpose
    plt.subplot(nrows,ncols,digits+1,frameon=False)
    # Next Image of a train set
    next_digit = test[digits].reshape(28,28)
    # plt.imshow is used for plotting numerical values
    plt.imshow(next_digit,cmap = "gray")
# Padding for images. Because values overlaps
plt.tight_layout()
plt.show()

#test,predicted
#2   2
#0   0
#9  9
#0  0
#3  3
#7  7
#0  0
#3  3
#0  0
#3  3
## Loss Plots

val_plot = my_model.history.history
# validation loss
plt.plot(val_plot['val_loss'])
# train loss
plt.plot(val_plot['loss'])
plt.title('loss functions')

plt.legend(['val_loss', 'train_loss'], loc='upper left')
plt.show()

## Parameters selection with using Grid Search Algorithm
# Test #2
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

## Training model with result of grid search

# Test #2
learning_rate = 0.001
epochs = 38
batch_size = 100
METRICS = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name="recall")
            ]

#Assigning function
my_model = my_model(METRICS)

# Train the model on the training set.
epochs, hist = train_model(my_model,X_train, Y_train, epochs, batch_size)

my_model.evaluate(X_val,Y_val,verbose = 1, batch_size=batch_size)


## Calculating the Execution Time

end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))