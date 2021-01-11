import tensorflow as tf
import numpy as np
import pickle, os 
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras import layers, preprocessing
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, \
    Activation, Flatten
from tensorflow.keras.optimizers import SGD, Adam, Adadelta
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

tf.compat.v1.disable_eager_execution()

np.random.seed(40)  # for reproducibility
seed = 40
# important inputs to the model: 
batch_size = 32  # number of training samples used at a time to update the weights, was 5000
nb_classes = 2  # 10    # number of output possibilities: [0 - 9] KEEP
img_rows, img_cols = 224, 288 #153, 234  # 350, 230 #227, 227  #orign 700x460 # the size of the MNIST images was 28, 28
input_shape = (img_rows, img_cols)
nb_filters = 32    # number of convolutional filters to use, more filters => slower training
pool_size = (3, 3)  # pooling decreases image size, reduces computation, adds translational invariance
kernel_size = (5, 5)  # convolutional kernel size, slides over image to learn features aka filter size


class CNN(object):

    def __init__(self):
        """
        Initialize class with basic attributes

        Args:
        project_name (str): project name, used for saving models
        target_size (tuple(int, int)): size of images for input
        augmentation_strength (float): strength for image augmentation transforms
        batch_size(int): number of samples propogated throught network
        preprocessing(function(img)): image preprocessing function

        """


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
        print ('What do X_train, X_test, y_train, y_test look like {} {} {} {}'.  \
               format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
        print ('How many train ben/malig {} out of total {}'.format( \
               self.y_train.sum(axis=0), \
               len(self.y_train)) )
        print ('How many test ben/malig {} out of total {}'.format( \
               self.y_test.sum(axis=0), \
               len(self.y_test)))

    
    def data_augment(self):
        '''
        Arguments: 
            is_save: whether to output augmented images for use by future comparison models
            aug_dir: filepath
        '''

        image_gen_train = preprocessing.image.ImageDataGenerator(
                                                                rotation_range=20, 
                                                                #featurewise_center=False,
                                                                #featurewise_std_normalization=True,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                horizontal_flip=True, 
                                                                vertical_flip=True,
                                                                fill_mode='wrap')
        # compute quantities required for featurewise normalization                                                      
        image_gen_train.fit(self.X_train, augment=True, rounds=5)

        self.datagen = image_gen_train.flow(self.X_train, self.y_train, shuffle=True)

        image_gen_val = preprocessing.image.ImageDataGenerator()
        #image_gen_val.fit(self.X_test)
        self.val_datagen = image_gen_val.flow(self.X_test, self.y_test, shuffle=True)


    def get_image_gen(self, quantity=5):
        '''
        gets some data augmented samples from training data
        Arguments:
            quantity (optional) = how many images to return from self.datagen (based on training data)
        Returns:
            tuple(numpy array of images,  predicted value)   
        '''

        arr = self.datagen(5)
        print(arr.shape)
        return arr

    def load_and_featurize_data(self):
        '''
        Arguments:
            is_augment: whether we are using original source data and doing our own data augmentation
            is_save: whether to save the generated data augmented images to aug_dir
            aug_dir: if is_augment is true, this is the output dir to save.
                if is_augment is false, this is the read dir 
            The purpose of saving augmented data to disk is to enable future runs and model comparisons
            to use identical data for training.
        '''

        # comment out but might need again
        #self.X_train = self.X_train.reshape(self.X_train.shape[0], img_rows, img_cols, 3)
        #self.X_test = self.X_test.reshape(self.X_test.shape[0], img_rows, img_cols, 3)

        # convert and normalization
        self.X_train = self.X_train.astype('float32')  # data was uint8 [0-255]
        self.X_test = self.X_test.astype('float32')    # data was uint8 [0-255]
        self.X_train /= 255.0  # normalizing (scaling from 0 to 1)
        self.X_test /= 255.0   # normalizing (scaling from 0 to 1)

        print('X_train shape:', self.X_train.shape)
        print(self.X_train.shape[0], 'train samples')
        print(self.X_test.shape[0], 'test samples')

        self.data_augment()


    def define_model(self, nb_filters, kernel_size, input_shape, pool_size):
        model = Sequential()  # model is a linear stack of layers 

        # L1
        model.add(Conv2D(nb_filters,
                    (kernel_size[0], kernel_size[1]),  # ex. 3x3 or 5x5
                    padding='same',  # other options incl 'valid'
                    activation='relu',
                    input_shape=input_shape))  # first conv. layer, expect input shape to be image shape e.g. 227x227x3?

        model.add(MaxPooling2D(pool_size=pool_size))  # decreases size, helps prevent overfitting
        model.add(Dropout(0.2))  # zeros out some fraction of inputs, helps prevent overfitting

        # l2
        model.add(Conv2D(nb_filters * 2,
                        (kernel_size[0], kernel_size[1]),
                        activation = 'relu',
                        padding='valid'))  # 2nd conv. layer KEEP

        model.add(MaxPooling2D(pool_size=pool_size))  # decreases size, helps prevent overfitting
        model.add(Dropout(0.2))  # zeros out some fraction of inputs, helps prevent overfitting

        # L3
        model.add(Conv2D(nb_filters * 4,
                        (kernel_size[0], kernel_size[1]),
                        activation = 'relu',
                        padding='valid'))  

        model.add(MaxPooling2D(pool_size=pool_size))  # decreases size, helps prevent overfitting

        model.add(Flatten())  # necessary to flatten before going into conventional dense layer  KEEP
        print('Model flattened out to ', model.output_shape)

        # now start a typical neural network
        # Layer 1
        model.add(Dense(64, activation='relu'))  #         
        # model.add(Dropout(0.2))  # zeros out some fraction of inputs, helps prevent overfitting

        # Layer 2
        model.add(Dense(64, activation='relu'))  #         
        # model.add(Dropout(0.2))  # zeros out some fraction of inputs, helps prevent overfitting

        model.add(Dense(nb_classes, activation='softmax'))  # 10 final nodes (one for each class)  

        self.model = model

    def train_model(self, batch_size=32, epochs=10, verbose=1, data_augmentation=True, data_multiplier=1 ) :
        '''
        Arguments:  note: all default values are super low to not train well but not runaway; really you should set them
            batch_size int (32 is good)
            epochs int recommend > 50
            verbose int
            data_augmentation bool recommend True for live use
            brief_mode = True for small subset of training images, for debugging 
        This method lines up the model fit, either with X and y data, or with data augmentation added.
        Note: Standardization is handled outside this, in setting up self.X_train, y_train, X_test, and y_test
        
        '''
        print ('in fit_model, where we set up data aug which is set to {}'.format(data_augmentation))


        tbCallBack = TensorBoard(log_dir='../logs', histogram_freq=0, write_graph=True, write_images=True, \
                                 update_freq='epoch')

        # Todo: This was not working! (superbug)
        # modelCheckPtCallback = ModelCheckpoint(filepath='../models/best_model.hdf5',
        #                    save_best_only=True)

        if not data_augmentation:
            self.history = self.model.fit(self.X_train, self.y_train, 
                                          batch_size=batch_size, 
                                          epochs=epochs, 
                                          verbose=1,
                                          validation_data=(self.X_test, self.y_test)
                                          )
        else:
            print('Using datagen ') 
            self.history = self.model.fit(self.datagen, 
                                          epochs=epochs, 
                                          verbose=1,
                                          validation_data=self.val_datagen,
                                          steps_per_epoch=(len(self.X_train) // batch_size) * data_multiplier,
                                          callbacks=[tbCallBack]   #modelCheckPtCallback
                                          )
            print ('After self.model.fit() using datagen and steps_per_epoch')                                                         

    def compile_model (self, optimizer_name='SGD', lr=None):

        # todo: add kwargs to this method 
        # options to try with tuning
        optimizer_sgd = SGD(learning_rate=1e-5, momentum=0.0, nesterov=False, name='SGD')  # default learning = 0.01
        optimizer_adam = Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam') # 0.001
        optimizer_adadelta = Adadelta(learning_rate=lr, rho=0.95, epsilon=1e-07, name='Adadelta')  # if lr is None, will the default be used?

        '''
        loss_function = tf.keras.losses.CategoricalCrossentropy( \
                        from_logits=True, label_smoothing=0, reduction=losses_utils.ReductionV2.AUTO,
                        name='categorical_crossentropy')
        '''                

        self.model.compile(loss='sparse_categorical_crossentropy',   # 
                    optimizer='Adadelta',  # adapts learning rates based on a moving window of gradient updates, ...
                    # instead of accumulating all past gradients. This way, Adadelta continues learning even when many updates have been done.
                    metrics=['accuracy'])  # we might prefer to use F1, Precision, or sparse_categorical_crossentropy, crossentropy

        print ('model \n {}'.format(self.model.summary() ))   


    def save_model_new_format(self, save_dir, model_name):
        # Save model and weights, in SavedModel format.  Not in used b/c having trouble re-loading
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
        tf.keras.models.save_model(self.model, model_path, overwrite=True, include_optimizer=True)
        print('Saved trained model at %s ' % model_path)    

    def save_model_full(self, model_path, label, n_epochs, learning_rate):
        '''
        Arguments:
            model_path string
            label: string use for any differentiating labels on hyperparameters
            n_epochs: int
            learning_rate: float

        '''
        directory = os.path.dirname(model_path)
        if not os.path.exists(directory):
            os.mkdir(directory)

        model_name = "cnn_{}_epochs{}_lr{}".format(label, n_epochs, learning_rate)
        if model_path[-1] != "/":
            model_path = model_path + "/"
        tf.keras.models.save_model(self.model, model_name, overwrite=True, include_optimizer=True)

if __name__ == '__main__':

  print ('Hi, you should include me in bc.py and execute from there')