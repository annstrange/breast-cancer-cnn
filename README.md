# breast-cancer-cnn
Breast Cancer biopsy image analysis using CNN

# Objective
The goal of this project is to train and tune an artificial network to accurately classify breast cancer biopsies as malignant or benign. 

# Data Source




# EDA and how to run
To run this yourself, 
1. Fork/clone this github
2. download and unzip the data source so the folder structure looks like this
!(image of folder structure)[]

3. from /src run python bc.py
Options: To change hyperparameters, edit the global variables in bc.py

# Convolving
Using Scikit skimage which is good for feature detection, filtering, contour models, morphology, and classification problems.

Starting set of slides (one for each type of tumor)
!(Initial images)

- Choose greyscale or ID prominent colors?  There is significance between e.g. pink, blue, purple and brown because the HE dyes highlight
< description here >
Can we isolate the differences by identifying the prominent colors? 
Immunohistochemical staining colors separation:



- After identifying the outline
<images here>





# Training / Testing


# Model Selection


# Choice of Hypterparameters

1. Structure
2. Activation functions
    2a. ReLu (Rectified Linear Unit) - does well with positive numbers (true for images)
    2b. Softmax in the last layer.  When we want to next classify into the specific types of tumor (multiple classifications), this will facilitate multiple output classifications.
3. Weight and bias initialization
4. Training method
5. Regularization - yes.  We're going for a deeper model, and then use regularization to reduce overfit
6. Weight decay
7. Random seed - we should try some different ones to help detect/avoid reaching a local minimum in our Gradient descent within our model, used for optimizing the parameters by the model.

* Loss function - <formula here>
* Gradient Descent method:  Batch/Mini-batch/Stochastic.  We're using stochastic 
   Optimizer = keras.io/optimizer of SGD for Stochastic Gradient Descent.  Our activation function has a derivative at all points, so this should perform well. 


Table of Cost Measures with different parameters:





