import cv2
import pickle
import progressbar
import numpy as np
import pandas as pd
import seaborn as sns
from time import time
import matplotlib.pyplot as plt

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import BatchNormalization
from tensorflow.keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

class CNN:

    def __init__(self,x_train,y_train,x_test,y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def set_architecture(self,structure,batch_normalization=False,activation='relu',learning_optimizer='rmsprop'):
        self.model = Sequential()

        if(structure[0][0]=='conv'):
            self.model.add(Conv2D(structure[0][1], (3, 3), padding='same',input_shape=self.x_train.shape[1:]))
            self.model.add(Activation(activation))
            if batch_normalization: self.model.add(BatchNormalization())

        for layer_type, size in structure[1:]:
            if(layer_type == 'conv'):
                self.model.add(Conv2D(size, (3, 3)))
                self.model.add(Activation(activation))
                if batch_normalization: self.model.add(BatchNormalization())

            if(layer_type == 'pool'):
                self.model.add(MaxPooling2D(pool_size=(size, size)))
                self.model.add(Dropout(0.25))

            if(layer_type == 'dense'):
                num_classes=10
                self.model.add(Flatten())
                self.model.add(Dense(size))
                self.model.add(Activation(activation))
                self.model.add(Dropout(0.5))
                self.model.add(Dense(num_classes))
                self.model.add(Activation('softmax'))

        if(learning_optimizer=='rmsprop'):
            # initiate RMSprop optimizer
            opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
        elif(learning_optimizer=='adagrad'):
            # initiate RMSprop optimizer
            opt = keras.optimizers.Adagrad(learning_rate=0.0001, decay=1e-6)
        elif(learning_optimizer=='adam'):
            # initiate RMSprop optimizer
            opt = keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6)

        # Let's train the model using RMSprop
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )

    def compile(self,batch_size,epochs,call_back):
        self.history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.x_test, self.y_test),
            shuffle=True,
            callbacks=[call_back]
    )

    def evaluate(self):
        # Score trained model.
        scores = self.model.evaluate(self.x_test, self.y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])


class TimingCallback(Callback):
    def __init__(self):
        self.logs=[]

    def on_epoch_begin(self,epoch, logs={}):
        self.starttime=time()

    def on_epoch_end(self,epoch, logs={}):
        self.logs.append(time()-self.starttime)
