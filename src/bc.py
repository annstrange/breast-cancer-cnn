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
from bc_plotting import *
from test_methods import *
#from boto3_conn import *

# Global variables
nb_epoch = 5
brief_mode = False  # use to take an even sub-sample for debugging; makes sure to hit all classes. 
#root_dir = '../data/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB'
# Note: EC2 has different struct. ../data/BreaKHist_v1 ...
root_dir = '../data/BreaKHis_v1/histology_slides/breast'
data_multiplier = 1
#root_dir = '../BreaKHis_v1/histology_slides/breast'

# Careful: outside numpy we would say this is a LxW shape
image_size = tuple((153, 234, 3))

def read_images(root_dir, sub_dirs= ['all']): 
	'''
	Input: None
	Output: ImagePipline Object

	Initialize the ImagePipleine object and read through all of the images 
	to attach them to our ImagePipeline object. 
	'''
	ip = ImagePipeline(root_dir)
	ip.read(sub_dirs, brief_mode=brief_mode)  # only pick one mag at a time, use brief_mode for debugging
	return ip

def test_sizes(ip, init=False):
	'''
	Input: ImagePipeline Object
	Output: None

	Run through a number of different sizes of pictures and pick which one fits best -
	i.e. strikes a good compromise between the size and resolution of the image. 
	''' 

	if init: 
		shapes = [(200, 200, 3), (200, 250, 3), (250, 300, 3), (300, 300, 3), (400, 400, 3)]
	shapes = [(100, 100, 3), (150, 150, 3)]

	for shape in shapes: 

		ip.resize(shape=shape)
		ip.show('40X', 0)
		ip.show('40X', 3)
		ip.show('40X', 6)
		ip.show('40X', 9)

	'''
	I can still tell that the images are cats and dogs with 100 pixels, so I'm going to try 
	starting with that and seeing how that turns out. I don't think it will end up well, 
	but I'd like to try it. 
	'''


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
	img = rgb2gray(img)
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
	
def run_pipeline():
	'''
	Performs steps to load files into ImagePipeline object
	'''


	ip = read_images(root_dir, ['200X'])
	#ip = read_images(root_dir)  # for all by default, this is heavy
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

def _group_split_holdouts(features, labels, groups, holdout_pct):
	'''
	Arguments:
		features: numpy array of images
		labels: numpy array of labels e.g 1/0
		groups: patient identifiers to keep groups together
	Returns:
		train_features,
		train_labels,
		train_groups,
		train_filenames,
		X_holdout,
		y_holdout
		[y_groups]: might be useful later, e.g. if 4 slides/5 for the same patient have same predict
		[y_filenames]: ditto
	'''
	pass	


def _groups_from_filenames(filenames, groups):
	'''
	Arguments:
		filenames: list of filenames any length
		attribs: full dictionary of attributes
	returns:
		list of len(filenames) with string slide-ids aka patient ids.
	'''
	groups = []


	return groups	

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

	# These are not random! 
	train_indx, test_indx = next(
			GroupShuffleSplit(random_state=0, test_size=holdout_pct).split(X, y, groups)
	)	
	X_train, X_hold, y_train, y_hold = \
		    X[train_indx], X[test_indx], y[train_indx], y[test_indx]
	print( X_train.shape, X_hold.shape)   # ((6,), (2,))
	# print('unique groups {} {}'. format( np.unique(groups[train_indx]), np.unique(groups[test_indx])))  # (array([1, 2, 4]), array([3]))
	# TypeError: only integer scalar arrays can be converted to a scalar index

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

	
def run_Kfolds(cnn, X_train, y_train, groups, filename_list, folds=3):
	'''
	Because our biopsies have many samples from the same patient, we should group these images together for splitting
	to avoid a leak (cheating)
	Arguments:
		X, y, attribs is data dictionary to get the patient id attribute for the groups. 
	Return:
		? X_train, X_test, y_train, y_test ?
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
	params = ['Adam']
	
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
			print ('About to call fit_model/start training')
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
			score_labeled = dict(zip(model.metrics_names, score))
			
		scores[i] = temp.mean()
	# which is the winner?	
	print(scores)  # [0.73862012 0.68346339 0.67930818] would indicate Adam
	
	# establish mean accuracy, recall, precision	


def execute_model(cnn, X_train, X_test, y_train, y_test):
	# This method assumes we've chosen our model and hyperparameters and are going for it.

	cnn.fit(X_train, X_test, y_train, y_test)
	cnn.load_and_featurize_data()

	# during fit process watch train and test error simultaneously
	print ('About to call fit_model')
	cnn.train_model( batch_size=32, epochs=nb_epoch,
				verbose=1, data_augmentation=True, data_multiplier=data_multiplier)

	cnn.save_model1('../', 'saved_model')

	plot_roc(X_test, y_test, cnn.model, 'roc_plot_cnn1')
	# get later with loaded_cnn = tf.keras.models.load_model('../cnn_model')_
	score = cnn.model.evaluate(X_test, y_test, verbose=1)

	y_pred = cnn.model.predict(X_test)
	print('predict results \n{}'.format(y_pred[:20]))

	#Taking argmax will tell the winner of each by highest probability. 
	y_pred_1D = np.argmax(y_pred, axis=-1).reshape(-1, 1)

	print (classification_report(y_test, y_pred_1D)) 

	print('Test scores:', score)
	print('Test accuracy:', score[1])  # this is the one we care about
	return cnn

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-n_epochs", type=int, help="Number of epochs for training", default=2)
	parser.add_argument("-data_multiplier", type=int, help="How much to expand data augmentation", default=1)
	'''
	parser.add_argument("-num_workers", type=int, help="Number of workers to parse data, default =0)", default=0)
	parser.add_argument("-batch_size", type=int, help="Number of batches during trainging/testing", default=20)
	parser.add_argument("-learning_rate", type=float, help="Learning rate during training", default=0.05) 
	parser.add_argument("-embedding_dim", type=int, help="Embedding dimention for training", default=300)
	parser.add_argument("-model_path", help="Path to save your model", default="/home/ec2-user/projects/models/")    


	main(args.num_workers, args.batch_size, args.n_epochs, args.learning_rate, args.embedding_dim, args.model_path)
	'''
	args = parser.parse_args()
	nb_epoch = args.n_epochs
	data_multiplier = args.data_multiplier

	# Boto S3 connection test want annstrange-cnn-boto3 
	# todo add try catch, but this totally works if env vars passed to docker run cmd. 
	#boto3_connection = get_s3('us-west-2')
	#if boto3_connection:
	#	print_s3_buckets_boto3(boto3_connection)

    # Clear any tensorboard logs from previous runs
	subprocess.call(["rm", "-rf", "../logs/"])

	ip = run_pipeline() # sets ip images_filename_list, and images_list
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

	# Useful for EDA
	img_dict = ip.get_one_of_each('M')
	#plot_images(img_dict)

	features = ip.features
	target = ip.tumor_class_vector

	# shuffle! 
	X, y, groups, filename_list = shuffle_all(ip.features, ip.tumor_class_vector, ip.group_list, ip.images_filename_list)

	# check the validity of shuffle
	print('----------check after shuffle----------------------')
	num_diffs = test_integrities(y, groups, filename_list, ip.images_attributes)
	print('-----------after shuffle, are {} diffs---------------------'.format(num_diffs))

	print ('shuffled!')
	# get train/test split while keeping slide-ids grouped together, to isolate holdouts
	X_train, X_holdout, y_train, y_holdout, groups_tr, groups_val, filename_tr, filename_val  = \
			train_holdouts_split_by_group(X, y, \
			groups=groups, filename_list=filename_list, holdout_pct=0.1)

	print ('after train_holdouts_split')

	# check the validity of shuffle
	print('--------------------------------')
	num_diffs = test_integrities(y_train, groups_tr, filename_tr, ip.images_attributes)
	print('-----------after train_holdouts_split, are {} diffs---------------------'.format(num_diffs))

	# initialize model
	cnn = CNN()
	cnn.define_model(nb_filters, kernel_size, image_size, pool_size)

	#cnn.fit(X_train, X_test, y_train, y_test)
	#cnn.load_and_featurize_data()

	# run/cross validation, how to we get model selection?  
	# expect to be sending in about 2371 records from 2636
	run_Kfolds(cnn, X_train, y_train, groups=groups_tr, filename_list=filename_tr, folds=3)

	cnn.save_model1('../', 'saved_model_adam')
	# With winning model(s), send validation data thru and get predict metrics
	# todo: set acutall winning hypteparameters on the cnn 
	# todo: can we just keep the winner from our evaluations? suspect 'best model' w be it.
	#execute_model(cnn, X_train, X_holdout, y_train, y_holdout)	


	if (cnn.model.history is not None):
		plot_training_results(history = cnn.model.history, epochs=nb_epoch)
	else:
		print ('finish without history')	
	
