'''Trains a simple convnet on the MNIST dataset.
based on a keras example by fchollet
Find a way to improve the test accuracy to almost 99%!
FYI, the number of layers and what they do is fine.
But their parameters and other hyperparameters could use some work.
'''
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from tensorflow.keras.layers import Conv3D, MaxPool3D
#from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import numpy as np
import pickle 
np.random.seed(1337)  # for reproducibility
# important inputs to the model: 
batch_size = 50  # number of training samples used at a time to update the weights, was 5000
nb_classes = 2 # 10    # number of output possibilities: [0 - 9] KEEP
nb_epoch = 50       # number of passes through the entire train dataset before weights "final"
img_rows, img_cols = 227, 227   # the size of the MNIST images was 28, 28
#input_shape_color = (img_rows, img_cols, 3)   # 1 channel image input (color) 
input_shape = (img_rows, img_cols)
nb_filters = 32    # number of convolutional filters to use, more filters => slower training
pool_size = (3, 3)  # pooling decreases image size, reduces computation, adds translational invariance
kernel_size = (5, 5)  # convolutional kernel size, slides over image to learn features aka filter size

class CNN(object):

    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None

    def fit(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        print ('What do X_train, X_test, y_train, y_test look like {} {} {} {}'. format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
        #print ('What do X_train, X_test, y_train, y_test look like {} {} {} {}'. format(X_train[0,0], X_test[0,0], y_train[0], y_test[0]))

        #print(X_train.shape)

    @staticmethod
    def one_hot_encode(y):
        # Categorical data to be converted to numeric data (can be 'B' and 'M' or 0 and 1)
        outcomes = y

        # Use the following code if y contains categories
        ### Universal list of colors
        total_outcomes = [0, 1]

        ### map each color to an integer, 
        mapping = {}
        for x in range(len(total_outcomes)):
            mapping[total_outcomes[x]] = x

        # integer representation
        for x in range(len(outcomes)):
            outcomes[x] = mapping[outcomes[x]]

        one_hot_encode = to_categorical(outcomes)
        print('one_hot_encoded {}'.format(one_hot_encode[:5,:]))

        return one_hot_encode

    def load_and_featurize_data(self):
        # the data, shuffled and split between train and test sets
        #(X_train, y_train), (X_test, y_test) = mnist.load_data()

        # reshape input into format Conv2D layer likes starting w shape like (1358, 230187) where 230187 is image dimensions
        self.X_train = self.X_train.reshape(self.X_train.shape[0], img_rows, img_cols, 3)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], img_rows, img_cols, 3)

        # don't change conversion or normalization
        self.X_train = self.X_train.astype('float32')  # data was uint8 [0-255]
        self.X_test = self.X_test.astype('float32')    # data was uint8 [0-255]
        self.X_train /= 255  # normalizing (scaling from 0 to 1)
        self.X_test /= 255   # normalizing (scaling from 0 to 1)

        print('X_train shape:', self.X_train.shape)
        print(self.X_train.shape[0], 'train samples')
        print(self.X_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices (don't change)

        #self.Y_train = to_categorical(self.y_train, nb_classes)  # cool
        #self.Y_test = to_categorical(self.y_test, nb_classes)
        # in Ipython you should compare Y_test to y_test
        #return X_train, X_test, Y_train, Y_test
        #print('y after one hot encoding {}'.format(self.Y_test.shape))


    def define_model(self, nb_filters, kernel_size, input_shape, pool_size):
        model = Sequential()  # model is a linear stack of layers (don't change)

        #input_shape_color = self.X_train.shape
        #L1

        # Use for 2D but applies to color images as well (filter will be 3-deep)
        model.add(Conv2D(nb_filters,
                    (kernel_size[0], kernel_size[1]),  # ex. 3x3
                    padding='same',  # other options incl 'valid'
                    input_shape=input_shape))  # first conv. layer, expect input shape to be image shape e.g. 227x227x3?
        model.add(Activation('relu'))  # Activation specification necessary for Conv2D and Dense layers

        model.add(MaxPooling2D(pool_size=pool_size))  # decreases size, helps prevent overfitting
        model.add(Dropout(0.25))  # zeros out some fraction of inputs, helps prevent overfitting

        model.add(Conv2D(nb_filters * 2,
                        (kernel_size[0], kernel_size[1]),
                        padding='valid'))  # 2nd conv. layer KEEP
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=pool_size))  # decreases size, helps prevent overfitting
        model.add(Dropout(0.25))  # zeros out some fraction of inputs, helps prevent overfitting

        model.add(Conv2D(nb_filters * 4,
                        (kernel_size[0], kernel_size[1]),
                        padding='valid'))  # 2nd conv. layer KEEP
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=pool_size))  # decreases size, helps prevent overfitting
        model.add(Dropout(0.25))  # zeros out some fraction of inputs, helps prevent overfitting

        model.add(Flatten())  # necessary to flatten before going into conventional dense layer  KEEP
        print('Model flattened out to ', model.output_shape)

        # now start a typical neural network
        model.add(Dense(32))  # (only) 32 neurons in this layer, really?          
        model.add(Activation('relu'))

        model.add(Dropout(0.25))  # zeros out some fraction of inputs, helps prevent overfitting

        model.add(Dense(nb_classes))  # 10 final nodes (one for each class)  
        model.add(Activation('softmax'))  # softmax at end to pick between classes 0-9 

        # many optimizers available, see https://keras.io/optimizers/#usage-of-optimizers
        # tip loss at 'categorical_crossentropy' is good for a this multiclass problem,
        # and KEEP metrics at 'accuracy'
        # suggest limiting optimizers to one of these: 'adam', 'adadelta', 'sgd'
        self.compile_model(model, 'adam')


    def compile_model (self, model, optimizer):
        model.compile(loss='binary_crossentropy',   # can also use categorical_crossentroy but need one-hot labels
                    optimizer=optimizer,  # adam -> .62
                    metrics=['accuracy'])  # we might use F1

        print ('model \n {}'.format(model.summary() ))           
        self.model = model

if __name__ == '__main__':

    cnn = CNN()

    # Have to get X data from bc.py
    cnn.fit(X_train, X_test, y_train, y_test)
    cnn.load_and_featurize_data()

    cnn.define_model(nb_filters, kernel_size, input_shape, pool_size)

    # during fit process watch train and test error simultaneously
    cnn.model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1, validation_data=(X_test, y_test))

    score = cnn.model.evaluate(X_test, y_test, verbose=0)


    print('Test scores:'.format(score))
    print('Test accuracy:', score[1])  # this is the one we care about
