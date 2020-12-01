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
#from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

#from pipeline import 
from skimage.filters import sobel 
from skimage.feature._canny import canny
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV

from image_pipeline import ImagePipeline
from image_convolv import * 
from cnn import *


def read_images(root_dir): 
	'''
	Input: None
	Output: ImagePipline Object

	Initialize the ImagePipleine object and read through all of the images 
	to attach them to our ImagePipeline object. 
	'''
	ip = ImagePipeline(root_dir)
	ip.read((['200X']), brief_mode=False)  # only pick one mag at a time, us brief_mode for debugging
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
	transformations = [sobel, canny, denoise_tv_chambolle, denoise_bilateral]
	transform_labels = ['sobel', 'canny', 'denoise_tv_chambolle', 'denoise_bilateral']
	ip.resize((227, 227, 3))
	for i, transformation in enumerate (transformations): 
		ip.transform(transformation, {})
		#ip.savefig('samples/', 1, transform_labels[i])
		#ip.savefig('samples/', 2, transform_labels[i])
		#ip.savefig('samples/', 3, transform_labels[i])


	'''
    Summary of transformations 
	'''

def fit_rand_forest(image_size, transformation=None):
	'''
	Input: ImagePipeline Object, Tuple, List
	Output: List of floats. 

	Fit a random forest using the images in an ImagePipeline Object and a number of different
	transformations (holding the image size fixed), and output the accuracy score for identifying 
	the classes of images (dogs and cats). 
	''' 

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

def fit_best_model(parameters): 
	'''
	Input: None
	Output: Fitted Random Forest Model

	Return the best fitted Random Forest Model that we have used from above. 
	'''

	print ('**** in fit_best_model ***** ')
	root_dir = '../data/BreaKHis_v1/histology_slides/breast'
	image_size = (227, 227, 3)
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
	img = resize(img, (227, 227, 3))
	img = rgb2gray(img)
	img = np.ravel(img)

	return img


if __name__ == '__main__':
	root_dir = '../data/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB'
	root_dir = '../data/BreaKHis_v1/histology_slides/breast'

	# Todo: play around with this after we have a cost function
	# starting size is 460 x 700 but sometimes 456 x 700
	image_size = (227, 227, 3)

	ip = read_images(root_dir)
	ip.resize(shape = image_size)

	# Turns data into arrays
	ip.vectorize()
	ip.vectorize_y() 

	# Useful
	#img_dict = ip.get_one_of_each()
	#plot_images(img_dict)
	
	# ok with color? if b&w
	#gray_imgs = get_grayscale(img_dict)
	#sobel_imgs = apply_filter(gray_imgs, img_filter = sobel, save_title='sobel_imgs.png', show_bool = False)
	#canny_imgs = apply_filter(gray_imgs, img_filter = canny, save_title='canny_imgs.png', show_bool = False)


	# Image convolving
	# do we want to exclude white areas completely?

	# find clusters of similar colors
	#centroids = apply_KMeans(img_dict)
	# or ID the dye colors (s/be pretty similar results; we'll compare)
	#dye_colors = dye_color_separation_dict(img_dict)


	# Apply actual transformations to the feature set
	#ip.transform(dye_color_separation, {})

	features = ip.features
	target = ip.tumor_class_vector

	
	print('features shape: {} and ex {}'.format(features.shape, features[:2, :2]))
    #print('target labels shape: {} and ex {}'.format(target.shape, target[:2]))

	print('shapes of train test input {} {}'.format(features.shape, target.shape))
	X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=1)
	#print('shapes of X_train, X_test, y_train, y_test {} {} {} {}'.format(X_train, X_test, y_train, y_test))

	print ('What do X_train, X_test, y_train, y_test look like {} {} {} {}'. format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
	print ('What do X_train, X_test, y_train, y_test look like {} {} {} {}'. format(X_train[0,0], X_test[0,0], y_train[0], y_test[0]))


	# Option B - Don't flatten so much, go w 4D numpy array all the way through
	# Make sure train_test_split is ok
	# Vectorize B


	# Run through model
	'''
	params = {'n_estimators': [10, 100, 1000], 'max_depth': [8, 12, None]}
	best_rf, best_params, best_score = fit_best_model(params)

	print("\nBest estimator from the GridSearchCV:")
	print(best_rf)
	print("\nBest parameters from the grid search:")
	print(best_params)
	print("\nCross validated score of best estimator:")
	print(best_score)
	'''

	#accuracy = fit_rand_forest (image_size)
	#print ('accuracy of Rand Forest w no transforms {}'.format(accuracy))  

	cnn = CNN()
	# Have to get X data from bc.py, pass as numpy arrays
	# Need y data one hot encoded
	y_train = cnn.one_hot_encode(y_train)
	y_test = cnn.one_hot_encode(y_test)

	cnn.fit(X_train, X_test, y_train, y_test)
	cnn.load_and_featurize_data()

	cnn.define_model(nb_filters, kernel_size, image_size, pool_size)

	# during fit process watch train and test error simultaneously
	
	cnn.model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
				verbose=1, validation_data=(X_test, y_test))

	score = cnn.model.evaluate(X_test, y_test, verbose=0)

	print('Test scores:', score)
	print('Test accuracy:', score[1])  # this is the one we care about