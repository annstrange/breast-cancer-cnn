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
brief_mode = False  # use to take an even sub-sample for debugging; makes sure to hit all classes. 

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
# flake8: noqa

#from pipeline import
# These might need a higher scikit-learn version than 0.13 which AWS has 
#from skimage.filters import sobel 
#from skimage.feature._canny import canny
#from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.model_selection import GridSearchCV


from image_pipeline import ImagePipeline
from image_convolv import * 
from cnn import *
from bc_plotting import *

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
	#root_dir = '../data/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB'
	root_dir = '../data/BreaKHis_v1/histology_slides/breast'

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
	
def execute_model(X_train, X_test, y_train, y_test):
	# The Model
	cnn = CNN()
	cnn.fit(X_train, X_test, y_train, y_test)
	cnn.load_and_featurize_data()

	cnn.define_model(nb_filters, kernel_size, image_size, pool_size)

	# during fit process watch train and test error simultaneously
	print ('About to call fit_model')
	cnn.train_model( batch_size=32, epochs=nb_epoch,
				verbose=1, data_augmentation=True)

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

	ip = run_pipeline()
	perform_image_transforms(ip)


	# Turns data into arrays
	ip.vectorize()
	ip.double_the_benigns()  # Evens out the classes

	# Useful for EDA
	img_dict = ip.get_one_of_each('M')
	#plot_images(img_dict)

	features = ip.features
	target = ip.tumor_class_vector

	
	print('features shape: {} '.format(features.shape))
	print('shapes of train test input {} {}'.format(features.shape, target.shape))
	X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = .2, random_state=1)


	print ('What do X_train, X_test, y_train, y_test look like {} {} {} {}'. format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))

	cnn = execute_model(X_train, X_test, y_train, y_test)

	cnn.model.save('../cnn.keras')
	# or?
	save_dir = os.path.join(os.getcwd(), 'saved_models')
	model_name = 'keras_cifar10_trained_model.h5' # where to save model
	# cnn.save_model(...)

	if (cnn.history is not None):
		plot_training_results(history = cnn.history, epochs=3)
	
