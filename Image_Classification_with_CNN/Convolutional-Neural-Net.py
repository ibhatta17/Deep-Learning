import os
import warnings
import numpy as np
warnings.filterwarnings('ignore')

from keras.preprocessing.image import ImageDataGenerator

# Importing the Keras libraries and packages
from keras.models import Sequential # to initialize the neural network
from keras.layers import Convolution2D # for convolutional layers(convolution section)
from keras.layers import MaxPooling2D # for pooling purpose 
from keras.layers import Flatten # for flattening(converting matrix to feature vector)
from keras.layers import Dense # add hidden(fully connected) layer to artificial neural net

from keras.preprocessing import image

def image_preprocessing():
	'''
    Image augmentation to prevent overfitting. Otherwise we might get great accuracy in training set and 
    poor accuracy in test set
    we will use 'keras' documentation shortcuts(ready to use codes)
    go to https://keras.io/
    Causes of overfitting
    1. Small number of samples for training
    Since we have only 8000 images for training, we need to trick(using image augmentation) to overcome overfitting
    Image augmentation means trasforming(rotating, flipping, shifting, shearing etc) so that we will have diverse 
    images to train our model

    The following code is copied form the above website in Image Preprocessing section
	'''
	
    # ------------- Image augmentation part -------------
	
    train_datagen = ImageDataGenerator(
            rescale = 1./255, # making sure all the pixel values are between 0 and 1
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True)

    validation_datagen = ImageDataGenerator(rescale = 1./255)

    # To create Training set
    training_set = train_datagen.flow_from_directory('dataset/training_set',
            target_size = (128, 128), # target image size
            batch_size = 32,
            class_mode = 'binary' # specifies number of dependent variables
    ) 

    # To create Validation set
    validation_set = validation_datagen.flow_from_directory('dataset/validation_set',
                                                            target_size = (128, 128), # target image size
                                                            batch_size = 32,
                                                            class_mode = 'binary' # specifies number of dependent variables
                                                           )
    return training_set, validation_set

def build_tunned_classifier():
    # Initializing the CNN
    classifier = Sequential()

    # -------------------------- Step 1 - Convolution -------------------------- 
    '''
	Convoltuion is required for feature extraction purpose. We use variety of feature detectors(filters) 
	in this step to find a filter which extracts the most of the features from an image. Lets try 32 such detectors.
    These filters are updated during the training to extract the best possble features
	'''
    classifier.add(Convolution2D(32, 3, 3, # number of feature detectors, size of feature detectors
                                 # 32 feature detectors each of size 3X3
    input_shape = (128, 128, 3), # since all the image are not of same size or format, we need to force them to 
                                 # have same format & size
    # since we are dealing with color image, we need 3D array, 64 X 64 is the dimension of the picture
    # in Tensorflow backend the order is size(dimension of 2D array) and then number of channel
    activation = 'relu' # rectifier activation function to make sure we do not have negative pixels to acheive non-linearity
    ))

    # -------------------------- Step 2 - Pooling -------------------------- 
    ''' 
	Pooling is required mainly to deal with spatial variance in the images and also to reduce the size of image with
    the features preserved from image. This will help to avoid overfitting to an extent and 
    obviously reduces the computational intensity
    Pooling with stride of 2 whereas the feature detection was done with the stride of 1 
	'''
	
    classifier.add(MaxPooling2D(pool_size = (2, 2) # 2 X 2 is a recommended pool size(hyperparameter candidate)
                               ))
    # Additing another convolutional layer to improve accuracy in test set
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    # for the 2nd layer(or any other layers onwards) we do not need to include input_shape
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Additing 3rd convolutional layer
    classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
    # Its a common practice to add double the feature detectors than in previous hidden layer. Hence 64 feature detectors
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # -------------------------- Step 3 - Flattening -------------------------- 
    classifier.add(Flatten())

    # -------------------------- Step 4 - Full Connection --------------------------  
    # Creating fully connected(hidden) layer
    classifier.add(Dense(output_dim = 128, activation = 'relu')) 
    '''
	output_dim should not be very small to make the classifier good model and not too hign to make sure 
    the model is computationally efficient. Its a common practice to choose power of 2
    since this a hidden layer, we use rectifier activation function
	'''
    
    classifier.add(Dropout(p = 0.1 # fraction of number of neurons that we want to disable to the total number of neurons
        ))
    '''
	It is recommended to start with p = 0.1 and then see if it solves the overfitting and increase the value of p accordingly
    We need to be aware that very high value of p will cause underfitting(more neurons will be disabled and the NN will not be able to learn enough)
	'''

    # Creating the output layer
    classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) 
    
    '''
	since this is the o/p layer, we use sigmoid activation function
    since the o/p is either dog or cat. i.e binary o/p, output_dim = 1
    For multi-class classification problem, we need to use 'softmax' activation function instead of 'sigmoid' 
    '''
	classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier

	
def compile_classifier():
    classifier = build_tunned_classifier()
    classifier.compile(optimizer = 'adam', # algorithm to find optimal set of weights for NN
                       # 'adam' is a stochastic gradient descent algorithm
    loss = 'binary_crossentropy', # loss function within the stochastic gradient descent algorithm
    # binary_crossentropy -> for binary o/p and categorical_crossentropy -> for categorical o/p
    metrics = ['accuracy'] # accuracy criterion to evaluate the model
    )
    
    return classifier

def train_classifier(training_set, validation_set):
	# To fit the convolution model in Training set and test in the Test set
	classifier = compile_classifier()
	classifier.fit_generator(training_set,
			steps_per_epoch = 8000,
			epochs = 25,
			validation_data = validation_set,
			validation_steps = 2000)
	return classifier
	
def make_prediction(classifier, image):    
    test_set = image.load_img('dataset/test_set/' + image, 
                             target_size = (128, 128))
    # since the input_shape of the image was 128X128 while we train our model, lets force the image to be of
    # same size here as well
    test_set = test_set.img_to_array(test_set) # converting 2D-image to 3D array
    # A new dimension is reqired to add to the 3D array to specify batch size
    test_set = np.expand_dims(test_set, axis = 0)

    # using the trained model for new prediction
    result = classifier.predict(test_set)
    
    # To identigy which class belongs to dog and cat
    d = training_set.class_indices
    for k,v in d.items():
        if result == val:
            prediction = k
            break
    return prediction

		
training_set, validation_set =  image_preprocessing()
classifier = train_classifier(training_set, validation_set)
pred = make_prediction(classifier, 'image1.jpg')
print('The image is predicted as: ', pred)
pred = make_prediction(classifier, 'image2.jpg')
print('The image is predicted as: ', pred)
