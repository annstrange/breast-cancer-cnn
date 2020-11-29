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
from image_pipeline import ImagePipeline
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

def read_images(root_dir): 
	'''
	Input: None
	Output: ImagePipline Object

	Initialize the ImagePipleine object and read through all of the images 
	to attach them to our ImagePipeline object. 
	'''
	ip = ImagePipeline(root_dir)
	ip.read((['40X']), brief_mode=True)  # only pick one mag at a time
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

def plot_images(imgs, save_title='imgs.png', show = False):
	'''
	Input: Dictionary of images, plotting options. 
	Output: Plot of 2-4 images.  
	'''
	if len(imgs) == 2:
		fig, axs = plt.subplots(1,2, figsize=(8, 8))
	elif len(imgs) == 4:
		fig, axs = plt.subplots(2,2, figsize=(8, 8))
	elif len(imgs) == 8:
		fig, axs = plt.subplots(4,2, figsize=(8, 8))
	else:
		raise ValueError("Plot function requires 2 or 4 items.")
	for ax, k, v in zip(axs.flatten(), imgs.keys(), imgs.values()): 
		ax.imshow(v, cmap='gray')
		ax.set_xticks([]); ax.set_yticks([])
		ax.set_title('Image ' + k)
	fig.savefig(save_title)
	if show: 
		plt.show()	

def test_transforms(ip): 
	'''
	Input: ImagePipeline Object
	Output: None

	Run through a couple of different transformations for our images and pick which one fits
	the best. 
	'''

    # instead of rgb2gray, find major colors
	transformations = [rgb2gray, sobel, canny, denoise_tv_chambolle, denoise_bilateral]
	transform_labels = ['rgb2gray', 'sobel', 'canny', 'denoise_tv_chambolle', 'denoise_bilateral']
	ip.resize((200, 300, 3))
	for i, transformation in enumerate (transformations): 
		ip.transform(transformation, {})
		ip.savefig('samples/40X', 1, transform_labels[i])
		ip.savefig('samples/40X', 2, transform_labels[i])
		ip.savefig('samples/40X', 3, transform_labels[i])


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

	print ('**** in fit_best_model ***** ')

	rf = RandomForestClassifier(random_state=1)
	root_dir = '../data/BreaKHis_v1/histology_slides/breast'
	image_size = (200, 300, 3)
	ip = read_images(root_dir)
	ip.resize(shape = image_size)

	if transformation == rgb2gray: 
		ip.grayscale()
	elif transformation == sobel: 
		ip.grayscale()
		ip.transform(sobel, {})
	
	ip.vectorize()
	ip.vectorize_y()
	features = ip.features

    # get vector of M/B for 
	target = ip.tumor_class_vector
	print('features shape: {} and ex {}'.format(features.shape, features[:2, :2]))
    #print('target labels shape: {} and ex {}'.format(target.shape, target[:2]))

	return

	X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=1)

	print ('What do X_train, X_test, y_train, y_test look like {} {} {} {}'. format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
	print ('What do X_train, X_test, y_train, y_test look like {} {} {} {}'. format(X_train[0,0], X_test[0,0], y_train[0], y_test[0]))


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
	image_size = (200, 300, 3)
	ip = read_images(root_dir)
	ip.resize(shape = image_size)

	ip.vectorize()
	features = ip.features
	# what does this look like?
	print('features shape: {} and ex {}'.format(features.shape, features[:2, :2]))

	target = ip.labels
	print('target labels shape: {} and ex {}'.format(target.shape, target[:2, :2]))


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
	img = resize(img, (200, 300, 3))
	#img = rgb2gray(img)
	img = np.ravel(img)

	return img


if __name__ == '__main__':
	root_dir = '../data/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB'
	root_dir = '../data/BreaKHis_v1/histology_slides/breast'

	# Todo: play around with this after we have a cost function
	# starting size is 460 x 700 but sometimes 456 x 700
	image_size = (200, 300, 3)

	ip = read_images(root_dir)
	ip.resize(shape = image_size)

	# Turns data into arrays
	ip.vectorize()
	ip.vectorize_y() 
	img_dict = ip.get_one_of_each()

	plot_images(img_dict)

	# Image convolving
	#params = {'n_estimators': [10, 100, 1000], 'max_depth': [8, 12, None]}
	#best_rf, best_params, best_score = fit_best_model(params)

	#print("\nBest parameters from the grid search:")
	#print(best_params)
	#print("\nCross validated score of best estimator:")
	#print(best_score)

