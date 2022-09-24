import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
data1=pd.read_csv("train.csv")
data=data1.values

#reshaping the dataset
x_train   = data[0:21000,1:]
x_train1  =x_train.reshape((21000,28,28))
y_train   = data[0:21000,0]
y_train1  =y_train.reshape((21000,))

x_test    = data[21000:, 1:]
x_test1   =x_test.reshape((21000,28,28))
y_test    = data[21000:, 0]
y_test1   =y_test.reshape((21000,))



#matplotlib inline # Only use this if using iPython
image_index = 77 # You may select anything up to 60,000
print(y_train1[image_index]) # The label is 8
plt.imshow(x_train1[image_index], cmap='Greys')


# Reshaping the array to 4-dims so that it can work with the Keras API
xtrain = x_train1.reshape(x_train1.shape[0], 28, 28, 1)
xtest = x_test1.reshape(x_test1.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Making sure that the values are float so that we can get decimal points after division
xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
xtrain /= 255
xtest /= 255
print('x_train shape:', xtrain.shape)
print('Number of images in x_train', xtrain.shape[0])
print('Number of images in x_test', xtest.shape[0])


# Importing the required Keras modules containing model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

# network training
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=xtrain,y=y_train1, epochs=10)


# network evaluation
model.evaluate(xtest, y_test1)

# test sample data
image_index = 234
plt.imshow(xtest[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(xtest[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())
