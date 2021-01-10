#!/usr/bin/env python3

import os
from glob import glob

import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras import layers, preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import TensorBoard
from sklearn.utils import shuffle

from build_transfer_model import create_transfer_model
from simple_cnn import create_model

np.random.seed(40)  # for reproducibility
seed = 40
# important inputs to the model: 
#batch_size = 32  # number of training samples used at a time to update the weights, was 5000
#nb_classes = 2  # 10    # number of output possibilities: [0 - 9] KEEP


class ClassificationNet(object):
    """Keras Image Classifier with added methods to create directory datagens and evaluate on holdout set
        """

    def __init__(self, project_name, target_size, preprocessing=None, batch_size=32):
        """
        Initialize class with basic attributes

        Args:
        project_name (str): project name, used for saving models
        target_size (tuple(int, int)): size of images for input
        augmentation_strength (float): strength for image augmentation transforms
        batch_size(int): number of samples propogated throught network
        preprocessing(function(img)): image preprocessing function

        """
        self.project_name = project_name
        self.target_size = target_size
        self.input_size = self.target_size + (3,) # target size with color channels
        self.train_datagen = ImageDataGenerator()
        self.validation_datagen = ImageDataGenerator()
        #self.train_generator = None
        #self.validation_generator = None
        self.batch_size = batch_size
        self.preprocessing = preprocessing
        self.class_names =  None
        self.history = None
        self.model = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_holdout = None
        self.y_holdout = None

    def _init_data(self, X_train, X_test, X_holdout, y_train, y_test, y_holdout):
        """
        Initializes class data
        Args:
            Train and test data in nparray formats.        
        """
        
        self.X_train = X_train
        self.X_test = X_test
        self.X_holdout = X_holdout
        self.y_train = y_train
        self.y_test = y_test
        self.y_holdout = y_holdout
        print ('What do X_train, X_test, y_train, y_test look like {} {} {} {}'. format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
        print ('How many train ben/malig {} out of total {}'.format(self.y_train.sum(axis=0), len(self.y_train)) )
        print ('How many test ben/malig {} out of total {}'.format(self.y_test.sum(axis=0), len(self.y_test)))

        # fix these
        self.nTrain = len(self.y_train) #: number of training samples
        self.nVal = len(self.y_test)  #: number of validation samples
        self.nHoldout = len(self.y_holdout) #: number of holdout samples
        self.n_categories = 2 #: number of categories, value_count()?
        self.class_names = self.set_class_names() #: text representation of classes


    def _create_generators(self):   # was data_augment
        '''
        Arguments: 
            is_save: whether to output augmented images for use by future comparison models
            aug_dir: filepath
        '''
        # convert and normalization
        self.X_train = self.X_train.astype('float32')  # data was uint8 [0-255]
        self.X_test = self.X_test.astype('float32')    # data was uint8 [0-255]
        #self.X_holdout = self.X_holdout.astype('float32')
        # traded for normalizing in data gen
        #self.X_train /= 255.0  # normalizing (scaling from 0 to 1)
        #self.X_test /= 255.0   # normalizing (scaling from 0 to 1)
        #self.X_holdout /= 255 

        print('X_train shape:', self.X_train.shape)
        print(self.X_train.shape[0], 'train samples')
        print(self.X_test.shape[0], 'test samples')

        image_gen_train = preprocessing.image.ImageDataGenerator(rotation_range=20, 
                                                                #zoom_range=10,
                                                                featurewise_std_normalization=True,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                horizontal_flip=True, 
                                                                vertical_flip=True,
                                                                fill_mode='wrap')
                                                            
        image_gen_train.fit(self.X_train, 
                            augment=True, 
                            rounds=5
                            )

        self.train_datagen = image_gen_train.flow(self.X_train, 
                                                  self.y_train, 
                                                  #target_size=self.target_size,
                                                  shuffle=True,
                                                  )

        image_gen_val = preprocessing.image.ImageDataGenerator( featurewise_std_normalization=True)

        self.validation_datagen = image_gen_val.flow(self.X_test, 
                                              self.y_test, 
                                              #target_size=self.target_size,
                                              shuffle=False,
                                              )

        """
        Create generators to read images from directory
        """

        '''
        From original code:

        # Set parameters for processing and augmenting images
        self.train_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocessing,
            rotation_range=15*self.augmentation_strength,
            width_shift_range=self.augmentation_strength,
            height_shift_range=self.augmentation_strength,
            shear_range=self.augmentation_strength,
            zoom_range=self.augmentation_strength
        )
        # no need for augmentation on validation images
        self.validation_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocessing
        )

        self.train_generator = self.train_datagen.flow_from_directory(
            self.train_folder,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True)

        self.validation_generator = self.validation_datagen.flow_from_directory(
            self.validation_folder,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False)
        '''

    def fit(self, X_train, X_test, X_holdout, y_train, y_test, y_holdout, model_fxn, optimizer, epochs, data_multiplier=1):
        """
        Fits the CNN to the data, then saves and predicts on best model

        Args:
            train_folder(str): folder containing train data
            validation_folder(str): folder containing validation data
            holdout_folder(str): folder containing holdout data
            model_fxn(function): function that returns keras Sequential classifier
            optimizer(keras optimizer): optimizer for training
            epochs(int): number of times to pass over data

        Returns:
            str: file path for best model
            """

        self._init_data(X_train, X_test, X_holdout, y_train, y_test, y_holdout)
        print(self.class_names)
        self._create_generators()
        model = model_fxn(self.input_size, self.n_categories)

        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Initialize tensorboard for monitoring
        #tensorboard = keras.callbacks.TensorBoard(
        #    log_dir=self.project_name, histogram_freq=0, batch_size=self.batch_size, write_graph=True, embeddings_freq=0)

        tbCallBack = TensorBoard(log_dir='../logs', histogram_freq=0, write_graph=True, write_images=True, \
                                 update_freq='epoch')    

        if not os.path.exists('models'):
            os.makedirs('models')

        # Initialize model checkpoint to save best model
        #savename = 'models/'+self.project_name+'.hdf5'
        #mc = keras.callbacks.ModelCheckpoint(savename, monitor='val_loss', 
        #                                     verbose=0, save_best_only=True, 
        #                                     save_weights_only=False, mode='auto',
        #                                     save_freq = 'epoch')
        #                                     # period=1) # deprecated

        self.history = model.fit(self.train_datagen,
                                      steps_per_epoch=self.nTrain/self.batch_size * data_multiplier,
                                      epochs=epochs,
                                      validation_data=self.validation_datagen,
                                      validation_steps=self.nVal/self.batch_size ,
                                      callbacks=[tbCallBack])


        print ('What does self.history look like {}'.format(self.history))
        # save just-fit model to compare
        model.save('../models/saved_xfer_model.h5')

        #best_model = load_model(savename)
        #self.model = best_model
        self.model = model


        #print('evaluating transferred best model')
        #accuracy = self.evaluate_model(best_model)
        return '../models/saved_xfer_model.h5'

        '''
        best_model = load_model(savename)

        # hooollld on, verify best_model is working 
        self.model = best_model
        self.model = model  # ending model, might not be best (but could be)
        print('evaluating best model')
        accuracy = self.evaluate_model(best_model)

        print('evaluating just-built final model')
        accuracy = self.evaluate_model(self.model)

        #todo: call history plot 

        return savename
        '''

    def evaluate_model(self, model, X_holdout=None, y_holdout=None):
        """
        evaluates model on holdout data
        Args:
            model (keras classifier model): model to evaluate
            holdouts assumed to have been set in fit(), but if not, can be set here too
        Returns:
            list(float): metrics returned by the model, typically [loss, accuracy]
            """

        # assume model is set
        if self.model is None:
            print ('weird: no model is set, ok will use the model you passed in')

        if X_holdout is not None:
            self.X_holdout = X_holdout    
        if y_holdout is not None:
            self.y_holdout = y_holdout  
            self.nHoldout = len(y_holdout)  
        	# Holdout scaled and reshaped?

        #print ('X_holdout (before rescale) values look like {}'.format(self.X_holdout[:1,:1, :1, :5]))
        #self.X_holdout = self.X_holdout.astype('float32')
        #self.X_holdout /= 255.0   # normalizing (scaling from 0 to 1)
        # does my holdout data look ok?
        print ('X_holdout values look like {}'.format(self.X_holdout[:1,:1, :1, :5]))    

        print ('shapes in evaluate_model X {} y {}'.format(self.X_holdout.shape, self.y_holdout.shape))    

        # not actally necessary
        image_gen_holdout = preprocessing.image.ImageDataGenerator(featurewise_std_normalization=True )

        self.holdout_datagen = image_gen_holdout.flow(self.X_holdout, 
                                              self.y_holdout, 
                                              #target_size=self.target_size,
                                              shuffle=False
                                              )

        metrics = model.evaluate(self.holdout_datagen,
                                           steps=self.nHoldout/self.batch_size,  
                                           #use_multiprocessing=True,
                                           verbose=1)
        print("holdout loss:")
        print(metrics[0])
        print("accuracy: ")
        print(metrics[1])
        return metrics

    def print_model_layers(self, model, indices=0):
        """
        prints model layers and whether or not they are trainable

        Args:
            model (keras classifier model): model to describe
            indices(int): layer indices to print from
        Returns:
            None
            """

        for i, layer in enumerate(model.layers[indices:]):
            print("Layer {} | Name: {} | Trainable: {}".format(i+indices, layer.name, layer.trainable))

    '''
    # todo: needs some repair to work
    def process_img(self,img_path):
        """
        Loads image from filename, preprocesses it and expands the dimensions because the model predict function expects a batch of images, not one image
        Args:
            img_path (str): file to load
        Returns:
            np.array: preprocessed image
        """
        original = load_img(filename, target_size = self.target_size)
        numpy_image = self.preprocessing( img_to_array(original))
        image_batch = np.expand_dims(numpy_image, axis =0)
        return image_batch
    '''
    '''
    def model_predict(self, img_path,model):
        """
        Uses an image and a model to return the names and the predictions of the top 3 classes

        Args:
            img_path (str): file to load
            model (keras classifier model): model to use for prediction

        Returns:
            str: top 3 predictions
            """
        im =  process_img(img_path)
        preds =  model.predict(im)

        # For when we have multiple classifications
        if (n_categories >= 3):
            top_3 = preds.argsort()[0][::-1][:3] # sort in reverse order and return top 3 indices
            top_3_names = class_names[top_3]
            top_3_percent = preds[0][[top_3]]*100
            top_3_text = '\n'.join([str(name) + str(percent) + '%' for name, percent in zip(top_3_names,top_3_percent)])
            return top_3_text
        else: 
            return preds
    '''

    def set_class_names(self):
        """
        Sets the class names, sorted by alphabetical order
        """
        #names = [os.path.basename(x) for x in glob(self.train_folder + '/*')]
        names = ['B', 'M']
        return sorted(names)


class TransferClassificationNet(ClassificationNet):
    """Image Classifier Implementing Transfer Methods"""

    def fit(self, X_train, X_test, X_holdout, y_train, y_test, y_holdout, model_fxn, optimizers, epochs, freeze_indices, warmup_epochs, data_multiplier):
        """
        Fits the CNN to the data, then saves and predicts on best model

        Args:
            train_folder(str): folder containing train data
            validation_folder(str): folder containing validation data
            holdout_folder(str): folder containing holdout data
            model_fxn(function): function that returns keras Sequential classifier
            optimizers(list(keras optimizer)): optimizers for training, first value is for warmup, second value is for training
            epochs(int): number of times to pass over data
            freeze_indices(list(int)): layer indices to freeze up to, first value is for warmup, second value is for training
            warmup_epochs(int): number of epochs to warm up head for

        Returns:
            str: file path for best model
            """
        self._init_data(X_train, X_test, X_holdout, y_train, y_test, y_holdout)
        self._create_generators()

        model = model_fxn(self.input_size, self.n_categories)
        self.change_trainable_layers(model, freeze_indices[0])
        # same as saying base_model.trainable = False  
        # model.trainable = False  # also says not to update mean and variance stats; keeps base model in inference mode

        model.compile(optimizer=optimizers[0],
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Initialize tensorboard for monitoring
        #tensorboard = keras.callbacks.TensorBoard(log_dir=self.project_name, 
        #                                          histogram_freq=0, 
        #                                          batch_size=self.batch_size, 
        #                                          write_graph=True, 
        #                                          embeddings_freq=0,
        #                                          update_freq='epoch')
        tbCallBack = TensorBoard(log_dir='../logs', histogram_freq=0, write_graph=True, write_images=True, \
                            update_freq='epoch')                                             

        if not os.path.exists('models'):
            os.makedirs('models')

        # Initialize model checkpoint to save best model
        #savename = 'models/'+self.project_name+'.hdf5'
        #mc = keras.callbacks.ModelCheckpoint(savename, monitor='val_loss', 
        #                                     verbose=0, save_best_only=True, 
        #                                     save_weights_only=False, mode='auto',
        #                                     save_freq = 'epoch')
        #                                     #period=1)

        self.history = model.fit(self.train_datagen,
                                      steps_per_epoch=self.nTrain/self.batch_size * data_multiplier,
                                      #epochs=warmup_epochs,
                                      epochs=warmup_epochs+epochs,
                                      validation_data=self.validation_datagen,
                                      validation_steps=self.nVal/self.batch_size,
                                      callbacks=[tbCallBack])

        print ('history from model.fit looks like {}'.format(self.history))                             

        #self.change_trainable_layers(model, freeze_indices[1])
        #model.compile(optimizer=optimizers[1], loss='sparse_categorical_crossentropy',
        #              metrics=['accuracy'])
        #self.history = model.fit(self.train_datagen,
        #                              steps_per_epoch=self.nTrain/self.batch_size * data_multiplier,
        #                              epochs=epochs,
        #                              validation_data=self.validation_datagen,
        #                              validation_steps=self.nVal/self.batch_size,
        #                              callbacks=[tbCallBack])
        # save just-fit model to compare

        model.save('../models/saved_xfer_model.h5')

        #best_model = load_model(savename)
        #self.model = best_model
        self.model = model

        #print('evaluating transferred best model')
        #accuracy = self.evaluate_model(best_model)
        return '../models/saved_xfer_model.h5'

    def change_trainable_layers(self, model, trainable_index):
        """
        unfreezes model layers after passed index, freezes all before

        Args:
        model (keras Sequential model): model to change layers
        trainable_index(int): layer to split frozen /  unfrozen at

        Returns:
            None
            """

        for layer in model.layers[:trainable_index]:
            layer.trainable = False
        for layer in model.layers[trainable_index:]:
            layer.trainable = True


def main():
    #train_folder = 'data/train_small'
    #validation_folder = 'data/validation_small'
    #holdout_folder = 'data/holdout_small'

    print ('data setup is needed => call from bc.py')
'''
    target_size = (100, 100)  # 299,299 is suggested for xception but is quite taxing on cpu
    epochs = 5
    batch_size = 32

    model_fxn = create_model
    opt = RMSprop(lr=0.001)

    simple_cnn = ClassificationNet('simple_class_test', target_size, 
                                   preprocessing=preprocess_input, batch_size=batch_size)
    simple_cnn.fit(train_folder, validation_folder, holdout_folder, model_fxn, opt, epochs)

    model_fxn = create_transfer_model
    freeze_indices = [132, 126] # first unfreezing only head, then conv block 14
    optimizers = [RMSprop(lr=0.0006), RMSprop(lr=0.0001)] # keep learning rates low to keep from wrecking weights

    warmup_epochs = 5
    epochs = epochs - warmup_epochs
    transfer_model = TransferClassificationNet('transfer_test', target_size, 
                                                augmentation_strength=0.2, 
                                                preprocessing=preprocess_input, 
                                                batch_size=batch_size)
    transfer_model.fit(train_folder, validation_folder, holdout_folder, model_fxn,
                       optimizers, epochs, freeze_indices, warmup_epochs=warmup_epochs)


if __name__ == '__main__':
    main()

'''