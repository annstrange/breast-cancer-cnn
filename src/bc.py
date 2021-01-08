# main bc for biopsy classifier
'''
This python CNN uses Keras with Tensorflow, CIFAR10 for convolutional image pre-processing.

Inputs:
Expect you have downloaded the Breast Cancer image data from ___ and unzipped into a folder structure like:

data
    - BreaKHis_v1 with unzipped contents
        - histology_slides
            - breast
                - benign
                    - SOB
                        - adenosis
                        - fibroadenoma
                        - phyllodes_tumor
                        - tubular_adenoma
                - malignant
                    - SOB
                        - ductal_carcinoma
                        - lobular_carcinoma
                        - mucinous_carcinoma
                        - papillary_carcinoma
Within each of the lowest directories, the subdirectories fan out, 
with a subfolder for each patient sample, then one for each magnification
level

Because the filenames contain all the attributes we need, we'll read
recursively looking for all .png files found and parse the filenames
for attributes, including relative path

Outputs:
Results and graphs output to the Output folder

'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import GroupKFold
from random import random, shuffle
from operator import itemgetter

import os
import subprocess
import argparse
import pandas as pd
##  --- flake8: noqa

#from pipeline import
from skimage.filters import sobel 
from skimage.feature._canny import canny
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.model_selection import GridSearchCV  #?


from image_pipeline import ImagePipeline
from image_convolv import * 
from cnn import *
from class_model_xfer import *
from build_transfer_model import create_transfer_model
from simple_cnn import create_model
from bc_plotting import *
from test_methods import *
#from boto3_conn import *

# Global variables
nb_epoch = 5
# brief_mode = False  # use to take an even sub-sample for debugging; makes sure to hit all classes. 
root_dir = '../data/BreaKHis_v1/histology_slides/breast'
data_multiplier = 1
#root_dir = '../BreaKHis_v1/histology_slides/breast'

#image_size = tuple((153, 234, 3))
image_size = tuple((153, 234, 3)) #tuple((299, 299, 3))	 # preserve aspect ratio


def read_images(root_dir, sub_dirs= ['all'], brief_mode=False): 
	'''
	Input: None
	Output: ImagePipline Object

	Initialize the ImagePipleine object and read through all of the images 
	to attach them to our ImagePipeline object. 
	'''
	ip = ImagePipeline(root_dir)
	ip.read(sub_dirs, brief_mode=brief_mode)  # only pick one mag at a time, use brief_mode for debugging
	return ip


def test_transforms(ip): 
	'''
	Input: ImagePipeline Object
	Output: None

	Run through a couple of different transformations for our images and pick which one fits
	the best. 
	'''

    # instead of rgb2gray, find major colors
	transformations = [sobel, canny, denoise_tv_chambolle, denoise_bilateral, dye_color_separation]
	transform_labels = ['sobel', 'canny', 'denoise_tv_chambolle', 'denoise_bilateral', 'dye_color_separation']
	ip.resize(image_size)
	for i, transformation in enumerate (transformations): 
		ip.transform(transformation, {})
		#ip.savefig('samples/', 1, transform_labels[i])
		#ip.savefig('samples/', 2, transform_labels[i])
		#ip.savefig('samples/', 3, transform_labels[i])


	'''
    Summary of transformations 
	'''

'''
def fit_rand_forest(image_size, transformation=None):
	
	Input: ImagePipeline Object, Tuple, List
	Output: List of floats. 

	Fit a random forest using the images in an ImagePipeline Object and a number of different
	transformations (holding the image size fixed), and output the accuracy score for identifying 
	the classes of images (dogs and cats). 
	 

	print ('**** in fit_rand_forest ***** ')

	# pass these in from getting the best params
	rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=12, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=1,
                       verbose=0, warm_start=False)
	root_dir = '../data/BreaKHis_v1/histology_slides/breast'
 
	ip = read_images(root_dir)
	ip.resize(shape = image_size)

	if transformation == rgb2gray: 
		ip.grayscale()
	elif transformation == sobel: 
		ip.grayscale()
		ip.transform(sobel, {})
	elif transformation == denoise_bilateral:
		ip.tv_denoise()
	elif transformation == dye_color_separation:
		ip.transform(dye_color_separation, {})
	
	ip.vectorize()
	ip.vectorize_y()
	features = ip.features

    # get vector of M/B for 
	target = ip.tumor_class_vector
	#print('features shape: {} and ex {}'.format(features.shape, features[:2, :2]))
    #print('target labels shape: {} and ex {}'.format(target.shape, target[:2]))

	print('shapes of train test input {} {}'.format(features.shape, target.shape))
	X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=1)
	#print('shapes of X_train, X_test, y_train, y_test {} {} {} {}'.format(X_train, X_test, y_train, y_test))

	print ('What do X_train, X_test, y_train, y_test look like {} {} {} {}'. format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
	#print ('What do X_train, X_test, y_train, y_test look like {} {} {} {}'. format(X_train[0,0], X_test[0,0], y_train[0], y_test[0]))


	rf.fit(X_train, y_train)
	rf_preds = rf.predict(X_test)
	rf_accuracy = accuracy_score(y_test, rf_preds)

	return rf_accuracy
'''
'''
def fit_best_model(parameters): 
	#Input: None
	#Output: Fitted Random Forest Model

	#Return the best fitted Random Forest Model that we have used from above. 

	print ('**** in fit_best_model ***** ')
	root_dir = '../data/BreaKHis_v1/histology_slides/breast'
	#image_size = (227, 227, 3)
	ip = read_images(root_dir)
	#ip.resize(shape = image_size)
	#ip.transform(dye_color_separation, {})

	ip.vectorize()
	ip.vectorize_y()
	features = ip.features
	# what does this look like?
	#print('features shape: {} and ex {}'.format(features.shape, features[:2, :2]))

	target = ip.tumor_class_vector
	print('target tumor class vector shape: {} and ex {}'.format(target.shape, target[:5]))


	rf = RandomForestClassifier()
	clf = GridSearchCV(rf, parameters, n_jobs=-1, cv=3, scoring='accuracy',
	                   verbose=True)
	clf.fit(features, target)
	return clf.best_estimator_, clf.best_params_, clf.best_score_	
'''
def predict_img(model, filepath): 
	'''
	Input: Fitted model, Numpy Array
	Output: String

	Transform the image to fit the transformations we were using for our best random
	forest model (100, 100, 3), (not) grayscale and then predict whether the given image is 
	M (malignant) or benign (B). 
	'''
	img = imread(filepath)
	img = img_transform(img)
	prediction = model.predict(img)

	output = 'M' if prediction == 1 else 'B'
	output = 0

	return output

def img_transform(img): 
	'''
	Input: Numpy Array
	Output: Numpy Array

	Transform an image by reshaping it and also applying a grayscale to it. 
	'''
	img = resize(img, image_size)
	#img = rgb2gray(img)
	img = np.ravel(img)

	return img

def apply_premodel_transforms(transformation):
    # Apply image transformations to the numpy array of data, pre X/y train/test split, and pre-vectorize	

	if transformation == rgb2gray: 
		ip.grayscale()
	elif transformation == sobel: 
		ip.grayscale()
		ip.transform(sobel, {})
	elif transformation == denoise_tv_chambolle:
		ip.tv_denoise(weight=0.3)	
	elif transformation == dye_color_separation:
		ip.transform(dye_color_separation, {})
	
def run_pipeline(brief_mode=False):
	'''
	Performs steps to load files into ImagePipeline object
	'''


	ip = read_images(root_dir, ['200X'], brief_mode)
	#ip = read_images(root_dir)  # for all by default, this is heavy
	#ip.apply_square_crop()

	ip.resize(shape = image_size)
	return ip

def perform_image_transforms(ip):
	'''
	Todo: These need evaluation: which is better.
	'''
	pass
	# 1. No transforms - full color

	# 2. Grayscal denoise chambolle - these reduce the dimensions which the model needs to handle. 
	#apply_premodel_transforms(rgb2gray)
	#apply_premodel_transforms(denoise_tv_chambolle)

	# Using only hematoxylin
	#apply_premodel_transforms(dye_color_separation)

	# ok with color? if b&w
	#gray_imgs = get_grayscale(img_dict)
	#sobel_imgs = apply_filter(gray_imgs, img_filter = sobel, save_title='sobel_imgs.png', show_bool = False)
	#canny_imgs = apply_filter(gray_imgs, img_filter = canny, save_title='canny_imgs.png', show_bool = False)

	# Image convolving

	# find clusters of similar colors
	#centroids = apply_KMeans(img_dict)
	# or ID the dye colors (s/be pretty similar results; we'll compare)
	#dye_colors = dye_color_separation_dict(img_dict)

	# Apply actual transformations to the feature set
	#ip.transform(dye_color_separation, {})
	

def _sort_list_by_index(list1, index):
	'''
	Arguments:
		list is a list of items
		index: numpy int array of same length as list, in a different order e.g. (2, 0, 1)
	Returns:
		list sorted in same way a numpy mask filter works	
	'''
	#zip_list = list(zip(list1, index))
	#result = sorted(zip_list, key=itemgetter(1))
	#sorted_list, i = zip(*result)

	sorted_list = [list1[idx] for idx in index]

	return sorted_list


def shuffle_all(X, y, groups, filename_list):
	'''
	The cross validation functions aren't that shuffley, keep getting all the benigns up front in the training.
	'''	
	index = np.arange(len(filename_list))
	np.random.shuffle(index)

	X_shuffled = X[index]
	y_shuffled = y[index]
	groups_shuffled = _sort_list_by_index(groups, index)
	fn_shuffled = _sort_list_by_index(filename_list,index)

	return X_shuffled, y_shuffled, groups_shuffled, fn_shuffled


def train_holdouts_split_by_group(X, y, groups, filename_list, holdout_pct=0.1):
	'''
	Because our biopsies have many samples from the same patient, we should group these images together for splitting
	to avoid a leak (cheating)
	Arguments:
		X, y, attribs is data dictionary to get the patient id attribute for the groups. 
		groups: list of slide_ids = distinct by patient
		filenames: list, as the index to find slide attributes, this splits too
	Return:
		X_train, X_holdout, y_train, y_holdout, group_train, filename_train, ....
	
	'''

	print ('in train_holdouts_split_by_group, shapes of X and y, groups, and fns {} {} {} {}'.format(X.shape, y.shape, len(groups), len(filename_list)))
	print ('groups: {}'.format(groups[:50]))
	#groups = np.arange(len(y))  
	#print ('groups arange: {}'.format(groups[:100]))

	# These are not looking very random
	train_indx, test_indx = next(
			GroupShuffleSplit(random_state=0, test_size=holdout_pct).split(X, y, groups)
	)	
	X_train, X_hold, y_train, y_hold = \
		    X[train_indx], X[test_indx], y[train_indx], y[test_indx]
	print( X_train.shape, X_hold.shape)   # ((6,), (2,))
	# print('unique groups {} {}'. format( np.unique(groups[train_indx]), np.unique(groups[test_indx])))  # (array([1, 2, 4]), array([3]))

	# use same index on lists groups and filenames, watch: compress is not 0 based	
	# groups_tr = list(compress(groups, [x+1 for x in train_indx]))  # this way is incorrect
	groups_tr = _sort_list_by_index(groups, train_indx)
	groups_hold = _sort_list_by_index(groups, test_indx )
	filename_tr = _sort_list_by_index(filename_list, train_indx)
	filename_hold = _sort_list_by_index(filename_list,test_indx)

	print( len(groups_tr), len(groups_hold))   # ((6,), (2,))
	print ('sample holdout filenames: {}'.format(filename_hold[:20]))  # all benign? test for balance.
	print('What do train/hold return indexes look like train {} and holdout {}'.format(train_indx[:15], test_indx[:15]))

	return X_train, X_hold, y_train, y_hold, groups_tr, groups_hold, filename_tr, filename_hold

	
def run_Kfolds(cnn, X_train, y_train, groups, filename_list, folds=3, nb_epoch=nb_epoch, data_multiplier=data_multiplier):
	'''
	Because our biopsies have many samples from the same patient, we should group these images together for splitting
	to avoid a leak (cheating)
	Arguments:
		X, y, attribs is data dictionary to get the patient id attribute for the groups. 
	Return:
		which one was best?

	Repeats splits to get Train/test data, respecting keeping patient slides toghter (group feature), to test
	different hyperparameters and do cross validation for measures.
	Assumes holdout set is already removed.	
	'''

	#X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
	#y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
	print ('in Kfolds, shapes of X and y {} {}'.format(X_train.shape, y_train.shape))

	#groups = np.arange(y_train.shape[0])   # temp override for debugging

	gkf = GroupKFold(n_splits=folds)

	scores = np.zeros(3)

	# Set up some parameter tests here, ex. different learning rates? this could grow..  

	params = ['Adadelta']
	lr_params = [0.001, 1e-04, 1e-05]

	for i, k in enumerate(params):
		print ('next parameter to cv {}'.format(k))	
		#knn = KNeighborsClassifier(n_neighbors=k)
		temp = np.zeros(folds)
		for j, (train_index, val_index) in enumerate(gkf.split(X_train, y_train, groups=groups)):
			X_tr = X_train[train_index]
			y_tr = y_train[train_index]
			X_vl = X_train[val_index]
			y_vl = y_train[val_index]
			#print('What do GroupKFold return indexes look like train {} and test {}'.format(train_index[:5], val_index[:5]))
			
			# use same index on lists groups and filenames	
			groups_tr = _sort_list_by_index(groups, train_index)
			groups_val = _sort_list_by_index(groups, val_index)
			filename_tr = _sort_list_by_index(filename_list, train_index)
			filename_val = _sort_list_by_index(filename_list, val_index)

			print( len(groups_tr), len(groups_val))   # ((6,), (2,))
			#print ('validation filenames: {}'.format(filename_val))
			
			cnn.compile_model(optimizer_name=k)
			cnn.fit(X_tr, X_vl, y_tr, y_vl)
			cnn.load_and_featurize_data()
	
			# during fit process watch train and test error simultaneously
			print ('About to call fit_model/start training w epochs {} and data mux {}'.format(nb_epoch, data_multiplier))
			cnn.train_model( batch_size=32, epochs=nb_epoch,
				verbose=1, data_augmentation=True, data_multiplier=data_multiplier)

			# accuracy = cnn.model.score(X_vl, y_vl)  # RF sytax
			try:
				plot_roc(X_vl, y_vl, cnn.model, 'roc_plot_cnn{}{}'.format(k, i))
			except:
				'Plot_roc failed'	

			score = cnn.model.evaluate(X_vl, y_vl, verbose=1)
			print ('score from model.evaluate {}'.format(score))
			temp[j] = score[1]  # [0.98, 0.62]  validation loss/accuracy
			score_labeled = dict(zip(cnn.model.metrics_names, score))
			
		scores[i] = temp.mean()
	# which is the winner?	
	print(scores)  # [0.73862012 0.68346339 0.67930818] would indicate Adam
	
	# establish mean accuracy, recall, precision	
	print(score_labeled)

def execute_model(cnn, X_train, X_holdout, y_train, y_holdout, nb_epoch, data_multiplier):
	# This method assumes we've chosen our model and hyperparameters and are going for it.

	scores = np.zeros(3)
	# Set up winning parameter tests here, ex. different learning rates? this could grow..  
	optimizer = 'Adadelta'

	cnn.compile_model(optimizer_name=optimizer)
	cnn.fit(X_train, X_holdout, y_train, y_holdout)
	cnn.load_and_featurize_data()

	# during fit process watch train and test error simultaneously
	print ('About to call train_model with epochs {} mux {}'.format(nb_epoch, data_multiplier))
	cnn.train_model( batch_size=32, epochs=nb_epoch,
				verbose=1, data_augmentation=True, data_multiplier=data_multiplier)

	cnn.model.save('../models/saved_model.h5')

def evaluate_model(modelx, X_holdout, y_holdout, df_hold):
	'''
	Arguments:
		the model, and our holdout data
		df_hold has dataframe of existing attributes
	Return:
		dataframe of full holdout results	
	Run holdout data through predict to get metrics and generate ROC
	'''	

	# Holdout scaled and reshaped?


	score = modelx.model.evaluate(X_holdout, y_holdout, verbose=1)
	print ('score from model.evaluate {}'.format(score))

	try:
		modelname = modelx.project_name
		print ('roc_plot_' + modelx.project_name)
		plot_roc(X_holdout, y_holdout, modelx.model, 'roc_plot_' + modelx.project_name)
	except:
		'Plot_roc failed'	

	y_pred = modelx.model.predict_proba(X_holdout)
	print('predict results \n{}'.format(y_pred[:20]))

	#Taking argmax will tell the winner of each by highest probability. 
	y_pred_1D = np.argmax(y_pred, axis=-1).reshape(-1, 1)

	print (classification_report(y_holdout, y_pred_1D)) 

	print('Test scores:', score)
	print('Test accuracy:', score[1])  # this is the one we care about

	# get results
	df_prob = get_dataframe_w_predict(df_hold, y_pred)
	print('got df_prob {}'.format(df_prob.iloc[0]))

	return df_prob

def load_data_pipeline(brief_mode=False):
	'''
	Perform the initial data loads/image pipeline
	Args:
		brief_mode loads minimum data for debugging
	Returns:
		image pipeline	
	'''	
	ip = run_pipeline(brief_mode=brief_mode) # sets ip images_filename_list, and images_list
	perform_image_transforms(ip)

	# Turns data into arrays
	ip.vectorize() # sets ip features and tumor_class_vector

	print('-------Checking starting list integ -------------------------')
	test_integrities(ip.tumor_class_vector, ip.group_list, ip.images_filename_list, ip.images_attributes)
	print('--------------------------------')

	ip.double_the_benigns()  # Evens out the classes, stratify instead?

	print('----------check after double----------------------')
	num_diffs = test_integrities(ip.tumor_class_vector, ip.group_list, ip.images_filename_list, ip.images_attributes)
	print('-----------after double, are {} diffs---------------------'.format(num_diffs))

	return ip

def image_train_val_hold_split(ip):
	'''
	Do the image splits once, for all models to execute (not for Kfolds)
	'''	
	# shuffle! Keep filename and attributes lists in same order
	X, y, groups, filename_list = shuffle_all(ip.features, ip.tumor_class_vector, ip.group_list, ip.images_filename_list)

	# check the validity of shuffle
	print('----------check after shuffle----------------------')
	num_diffs = test_integrities(y, groups, filename_list, ip.images_attributes)
	print('-----------after shuffle, are {} diffs---------------------'.format(num_diffs))

	print ('shuffled!')
	# get train/test split while keeping slide-ids grouped together, to isolate holdouts
	X_train, X_holdout, y_train, y_holdout, groups_tr, groups_hold, filename_tr, filename_hold  = \
			train_holdouts_split_by_group(X, y, \
			groups=groups, filename_list=filename_list, holdout_pct=0.1)

	# Since holdouts doesn't get attached to datagen, apply any preprocessing, rescale
	X_holdout 

	print ('after train_holdouts_split')

	# check the validity of shuffle
	print('--------------------------------')
	num_diffs = test_integrities(y_holdout, groups_hold, filename_hold, ip.images_attributes)
	print('-----------after train_holdouts_split, are {} diffs---------------------'.format(num_diffs))

	################## For Directly Execute  ############
	print ('train/test split for training!')
	# get train/test split while keeping slide-ids grouped together, to isolate holdouts
	X_train, X_val, y_train, y_val, groups_tr, groups_val, filename_tr, filename_val  = \
			train_holdouts_split_by_group(X_train, y_train, \
			groups=groups_tr, filename_list=filename_tr, holdout_pct=0.2)

	print('--------------------------------')
	num_diffs = test_integrities(y_train, groups_tr, filename_tr, ip.images_attributes)
	print('-----------after train_test_split, are {} diffs---------------------'.format(num_diffs))		

	print ('after train_test_split')

	return X_train, X_val, X_holdout, y_train, y_val, y_holdout, groups_tr, groups_val, groups_hold, filename_tr, filename_val, filename_hold


def run_alex_ish_net (X_train, X_val, X_holdout, y_train, y_val, y_holdout, df_hold, nb_epoch, data_multiplier):
	'''
	Args:
		ip has the image data read in and vectorized
	Returns:
		data split into Train/validation/holdout with corresponding filename and attribute lookups, 
		respecting patient slide ids 	
	'''

	# initialize AlexNet-ish model
	cnn = CNN()
	cnn.define_model(nb_filters, kernel_size, image_size, pool_size)

	##################### KFOlds #############
	# for Kfolds, do not do second split
	#run_Kfolds(cnn, X_train, y_train, groups=groups_tr, filename_list=filename_tr, folds=3, nb_epoch, data_multiplier)

	execute_model(cnn, X_train, X_val, y_train, y_val,nb_epoch, data_multiplier )
	# execute_model(cnn, X_train, X_holdout, y_train, y_holdout, groups=groups_tr, filename_list=filename_tr)

	df_results = evaluate_model(cnn, X_holdout, y_holdout, df_hold)
	print ('Holdout evaluation results {}'.format(df_results.iloc[0]))

	#cnn.model.save('../models/saved_model.h5')
	# With winning model(s), send validation data thru and get predict metrics
	# todo: set acutal winning hypteparameters on the cnn 
	# todo: can we just keep the winner from our evaluations? suspect 'best model' w be it.
	#execute_model(cnn, X_train, X_holdout, y_train, y_holdout)	


	if (cnn.history is not None):
		plot_training_results(history = cnn.model.history, epochs=nb_epoch)
	else:
		print ('finish without history')	
    		

def transfer_model_main(X_train, X_val, X_holdout, y_train, y_val, y_holdout, target_size, epochs, batch_size, data_multiplier):
	'''
	Run a transfer model from Xception
	'''
	model_fxn = create_model
	opt = RMSprop(lr=0.001)

	simple_cnn = ClassificationNet('simple_class_test', target_size, 
									preprocessing=preprocess_input, batch_size=batch_size)

	# suspect we don't need this								
	#simple_cnn.fit(X_train, X_val, X_holdout, y_train, y_val, y_holdout, model_fxn, opt, epochs, data_multiplier=data_multiplier)

	model_fxn = create_transfer_model
	freeze_indices = [132, 126] # first unfreezing only head, then conv block 14
	optimizers = [Adadelta(lr=0.0006), Adadelta(lr=0.0001)] # [RMSprop(lr=0.0006), RMSprop(lr=0.0001)] # keep learning rates low to keep from wrecking weights
	#optimizers = [SGD(lr=0.00006), SGD(lr=0.00001)]

	warmup_epochs = 5
	epochs = epochs - warmup_epochs
	transfer_model = TransferClassificationNet('transfer', target_size, 
												preprocessing=preprocess_input, 
												batch_size=batch_size)
	transfer_model.fit(X_train, X_val, X_holdout, y_train, y_val, y_holdout, model_fxn,
						optimizers, epochs, freeze_indices, warmup_epochs=warmup_epochs, data_multiplier=data_multiplier)


	# evaluate_model(transfer_model, X_holdout, y_holdout)
	transfer_model.evaluate_model(transfer_model.model, X_holdout, y_holdout)

	return transfer_model

def get_dataframe(y_holdout, groups_hold, filename_hold, attribs):
	'''
	Arguments:
		y numpy arrays of same len
		y_pred: numpy array of prediction probabilities by class (top 3 if multiclass)
		groups_hold: list of slide-ids for each record in X and y
		filename_hold: list of filenames for each record in X and y
		attribs: dictionary of attributes with key = filename, also index for getting image from X array
	Returns:
		dataframe of the items with their attributes for plotting	
	'''	

	df1 = pd.DataFrame(zip(filename_hold, groups_hold, y_holdout))
	df1.reset_index(inplace=True)
	df1.columns = (['index', 'file', 'group', 'y'])

	df_att = pd.DataFrame.from_dict({(i): attribs[i]
                           for i in attribs.keys() },
                       orient='index')
	df_att.reset_index(inplace=True)
	df_att.columns = (['file', 'tumor_class', 'biopsy_procedure', 'tumor_type', 'year', 'slide_id', 'mag', 'seq', 'image_size'])

	df_merged = pd.merge(df1, df_att, on=['file', 'file'])

	return df_merged	


def get_dataframe_w_predict(df1, y_proba):
	'''
	Arguments:
		df1: dataframe from get_dataframe of holdout results
	Returns:
		df that includes the prediction results	
	'''	
	df_p = pd.DataFrame(list(y_proba))
	df_p["y_hat_p"]= df_p[[0, 1]].max(axis=1)
	df_p['y_hat'] = df_p[0].apply(lambda x: 1 if x< .5 else 0)

	df_p.reset_index(inplace=True)
	df_p.columns = (['index', 'p_0', 'p_1', 'y_hat_p', 'y_hat'])

	df_prob = pd.merge(df1, df_p, on = ['index', 'index'])

	return df_prob


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-n_epochs", type=int, help="Number of epochs for training", default=2)
	parser.add_argument("-data_multiplier", type=int, help="How much to expand data augmentation", default=1)
	parser.add_argument("-brief_mode", type=int, help="Set to true for basic compile check; min data", default=0)
	'''
  
	main(args.num_workers, args.batch_size, args.n_epochs, args.learning_rate, args.embedding_dim, args.model_path)
	'''
	args = parser.parse_args()
	nb_epoch = args.n_epochs
	data_multiplier = args.data_multiplier
	brief_mode = (args.brief_mode == 1)
	
	image_size =   tuple((153, 234, 3))  # preserve aspect ratio tuple((307,467, 3)) tuple((299, 299, 3))
	#cropped_size = tuple((299, 299, 3))

	# Load image data
	ip = load_data_pipeline(brief_mode)


	# Boto S3 connection test want annstrange-cnn-boto3 
	# todo add try catch, but this totally works if env vars passed to docker run cmd. 
	#boto3_connection = get_s3('us-west-2')
	#if boto3_connection:
	#	print_s3_buckets_boto3(boto3_connection)

    # Clear any tensorboard logs from previous runs
	subprocess.call(["rm", "-rf", "../logs/"])

	# Useful for EDA
	# img_dict = ip.get_one_of_each('M')
	# plot_images(img_dict)

	features = ip.features
	target = ip.tumor_class_vector

	X_train, X_val, X_holdout, y_train, y_val, y_holdout, groups_tr, groups_val, groups_hold, \
		filename_tr, filename_val, filename_hold = image_train_val_hold_split(ip)

	# build df of holdouts for plotting
	df_hold = get_dataframe(y_holdout, groups_hold, filename_hold, ip.images_attributes)
	print('got df_hold {}'.format(df_hold.iloc[0]))


	#################
	run_alex_ish_net (X_train, X_val, X_holdout, y_train, y_val, y_holdout, df_hold, nb_epoch, data_multiplier)
	




	# New stuff -------------Transfer model

	#target_size = (299,299)  # 299,299 is suggested for xception but is quite taxing on cpu
	# Todo: bugfix ValueError: Error when checking input: expected input_1 to have shape (100, 100, 3) but got array with shape (299, 299, 3)
	#epochs = 5
	batch_size = 32

	
	# tf_model = transfer_model_main(X_train, X_val, X_holdout, y_train, y_val, y_holdout, target_size, nb_epoch, batch_size, data_multiplier)

	# df_prob = evaluate_model(tf_model, X_holdout, y_holdout, df_hold)

