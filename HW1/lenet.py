import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import Callback
from keras import backend as K
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KERAS_BACKEND']='tensorflow'

class my_model(object):
    def __init__(self,batch_size,num_classes,epochs,learning_rate,img_rows,img_cols):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.learning_rate = learning_rate        
        self.img_rows, self.img_cols = img_rows, img_cols
        self.sgd = SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True) 

    def print_parameters(self):
        print('hyperparameter:')
        print('batch size:', self.batch_size)
        print('learning rate:', self.learning_rate )
        print('optimizrt: SGD')

    def load(self):
        # the data, split between train and test sets
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, self.img_cols, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.img_rows, self.img_cols, 1)

        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255
        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)

    def build(self):
        self.model = Sequential()
        self.model.add(Conv2D(filters=6, kernel_size=(5,5), padding='valid', input_shape=(28,28,1), activation='tanh'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='tanh'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Flatten())
        self.model.add(Dense(120, activation='tanh'))
        self.model.add(Dense(84, activation='tanh'))
        self.model.add(Dense(10, activation='softmax'))
        
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=self.sgd,
                    metrics=['accuracy'])

    def train(self):
        self.histories = Histories()
        self.model.fit(self.x_train, self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_data=(self.x_test, self.y_test),
            callbacks=[self.histories])
        self.model.save_weights('my_model.h5')
        

    
    def plot(self):         
        plt.plot(self.histories.losses)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('iteration')
        plt.show()
    def predict(self,image):
        self.x = img_to_array(image)
        self.x = np.expand_dims(self.x, axis=0)
        self.y = self.model.predict(self.x)
        return self.y

    def load_weight(self):
        self.model.load_weights('my_model.h5')

class Histories(Callback):

    def on_train_begin(self,logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))

