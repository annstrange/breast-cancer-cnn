import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers  
from tensorflow.keras import layers, preprocessing
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
tf.compat.v1.disable_eager_execution()


import numpy as np
import pickle 
np.random.seed(40)  # for reproducibility
seed = 40
# important inputs to the model: 
batch_size = 50  # number of training samples used at a time to update the weights, was 5000
nb_classes = 2 # 10    # number of output possibilities: [0 - 9] KEEP
nb_epoch = 10       # number of passes through the entire train dataset before weights "final"
img_rows, img_cols = 153, 234  # 350, 230 #227, 227  #orign 700x460 # the size of the MNIST images was 28, 28
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

        print ('How many train ben/malig {} out of total {}'.format(self.y_train.sum(axis=0), len(self.y_train)) )
        print ('How many test ben/malig {} out of total {}'.format(self.y_test.sum(axis=0), len(self.y_test)))

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

    def data_augment(self):

        '''
        tf.keras.preprocessing.image.NumpyArrayIterator(
            x, y, image_data_generator, batch_size=32, shuffle=False, sample_weight=None,
            seed=None, data_format=None, save_to_dir=None, save_prefix='',
            save_format='png', subset=None, dtype=None
        )
        tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False, samplewise_center=False,
            featurewise_std_normalization=False, samplewise_std_normalization=False,
            zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0,
            height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0,
            channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False,
            vertical_flip=False, rescale=None, preprocessing_function=None,
            data_format=None, validation_split=0.0, dtype=None
        '''

        image_gen_train = preprocessing.image.ImageDataGenerator(rotation_range = 20, 
                #featurewise_center=True,
                rescale = 1./255,
                #featurewise_std_normalization=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True, 
                vertical_flip=True)
        #np_iter = preprocessing.image.NumpyArrayIterator(self.X_train, self.y_train, datagen)
        image_gen_train.fit(self.X_train, augment=True, rounds = 5)

        self.datagen = image_gen_train.flow(self.X_train, self.y_train)

        image_gen_val = preprocessing.image.ImageDataGenerator(
                rescale = 1./255,
                #featurewise_std_normalization=True
                )
        image_gen_val.fit(self.X_test)
        self.val_datagen = image_gen_val.flow(self.X_test, self.y_test,
                shuffle=True)
                

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        #self.datagen.fit(self.X_train)

    def get_image_gen(self, quantity=5):
        '''
        gets some data augmented samples from training data
        Arguments:
            quantity (optional) = how many images to return from self.datagen (based on training data)
        Returns:
            tuple(numpy array of images,  predicted value)   
        '''

        arr = self.datagen(5)
        print (arr.shape)
        return arr

    def load_and_featurize_data(self):
        # the data, shuffled and split between train and test sets
        #(X_train, y_train), (X_test, y_test) = mnist.load_data()

        # reshape input into format Conv2D layer likes starting w shape like (1358, 230187) where 230187 is image dimensions
        # needed?
        self.X_train = self.X_train.reshape(self.X_train.shape[0], img_rows, img_cols, 3)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], img_rows, img_cols, 3)

        # convert and normalization
        self.X_train = self.X_train.astype('float32')  # data was uint8 [0-255]
        self.X_test = self.X_test.astype('float32')    # data was uint8 [0-255]
        #self.X_train /= 255  # normalizing (scaling from 0 to 1)
        #self.X_test /= 255   # normalizing (scaling from 0 to 1)

        print('X_train shape:', self.X_train.shape)
        print(self.X_train.shape[0], 'train samples')
        print(self.X_test.shape[0], 'test samples')

        # if augment option turned on
        self.data_augment()




    def define_model(self, nb_filters, kernel_size, input_shape, pool_size):
        model = Sequential()  # model is a linear stack of layers (don't change)

        #input_shape_color = self.X_train.shape
        #L1
        model.add(Conv2D(nb_filters,
                    (kernel_size[0], kernel_size[1]),  # ex. 3x3 or 5x5
                    padding='same',  # other options incl 'valid'
                    activation = 'relu',
                    input_shape=input_shape))  # first conv. layer, expect input shape to be image shape e.g. 227x227x3?

        #model.add(MaxPooling2D(pool_size=pool_size))  # decreases size, helps prevent overfitting
        #model.add(Dropout(0.05))  # zeros out some fraction of inputs, helps prevent overfitting

        #l2
        model.add(Conv2D(nb_filters * 2,
                        (kernel_size[0], kernel_size[1]),
                        activation = 'relu',
                        padding='valid'))  # 2nd conv. layer KEEP


        model.add(MaxPooling2D(pool_size=pool_size))  # decreases size, helps prevent overfitting
        #model.add(Dropout(0.05))  # zeros out some fraction of inputs, helps prevent overfitting

        #L3
        model.add(Conv2D(nb_filters * 4,
                        (kernel_size[0], kernel_size[1]),
                        activation = 'relu',
                        padding='valid'))  


        model.add(MaxPooling2D(pool_size=pool_size))  # decreases size, helps prevent overfitting
        #model.add(Dropout(0.05))  # zeros out some fraction of inputs, helps prevent overfitting

        model.add(Flatten())  # necessary to flatten before going into conventional dense layer  KEEP
        print('Model flattened out to ', model.output_shape)

        # now start a typical neural network
        # Layer 1
        model.add(Dense(64, activation = 'sigmoid'))  #         
        #model.add(Dropout(0.2))  # zeros out some fraction of inputs, helps prevent overfitting

        # Layer 2
        model.add(Dense(64, activation = 'sigmoid'))  #         
        #model.add(Dropout(0.2))  # zeros out some fraction of inputs, helps prevent overfitting

        model.add(Dense(nb_classes, activation = 'softmax'))  # 10 final nodes (one for each class)  

        # many optimizers available, see https://keras.io/optimizers/#usage-of-optimizers
        # tip loss at 'categorical_crossentropy' is good for a this multiclass problem,
        # suggest limiting optimizers to one of these: 'adam', 'adadelta', 'sgd'
        self.compile_model(model)

    def fit_model(self, batch_size=batch_size, epochs=nb_epoch, verbose=1, data_augmentation=False ) :
        '''
        This method lines up the model fit, either with X and y data, or with data augmentation added.
        Note: Standardization is handled outside this, in setting up self.X_train, y_train, X_test, and y_test
        
        '''

        if data_augmentation == False:
            self.history = self.model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1,
                validation_data = (self.X_test, self.y_test)
                )
        else:

            self.history = self.model.fit(self.datagen, 
                epochs=nb_epoch, 
                verbose=1,
                validation_data = self.val_datagen,
                validation_steps = 3)
                #max_queue_gen = 10 )   # default
                #steps_per_epoch=500, # (len(self.X_train) / batch_size, nb_epoch),
                #steps_per_epoch= (len(self.X_train) / batch_size, nb_epoch))
  
                

    def compile_model (self, model):

        # options to try with tuning
        optimizer_sgd = SGD(learning_rate=1e-5, momentum=0.0, nesterov=False, name='SGD')  # default learning = 0.01
        optimizer_adam = Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam') # 0.001

        model.compile(loss='sparse_categorical_crossentropy',   # can also use sparse_categorical_crossentropy 
                    optimizer='Adadelta',  # adapts learning rates based on a moving window of gradient updates, ...
                    # instead of accumulating all past gradients. This way, Adadelta continues learning even when many updates have been done.
                    metrics=['accuracy'])  # we might prefer to use F1, Precision, or SparseCategoricalAccuracy

        print ('model \n {}'.format(model.summary() ))   
     
        self.model = model


if __name__ == '__main__':

  print ('Hi, you should include me in bc.py and execute from there')