# Importing the libraries
import matplotlib.pyplot as plt
import tensorflow as tf

# Importing the MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)

# Visualizing the data
plt.imshow(mnist.train.images[1].reshape(28, 28)) # visualizing one of the handwritten digit
plt.show()

class Weight_Init:
    @staticmethod
    # for weigts
    def init_weights(shape):
        init_random_dist = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(init_random_dist)
    
    @staticmethod
    # for bias
    def init_bias(shape):
        init_random_dist = tf.constant(0.1, shape = shape)
        return tf.Variable(init_random_dist)


def convolution_layer(input_x, shape):
    '''
    Input:
        input_x: batch of input images [batch, height, weight, color channel]
        shape: 
    '''
    
    W = Weight_Init.init_weights(shape) # dimension: [filter H, filter W, Channel IN, channel OUT]
    b = Weight_Init.init_bias([shape[3]])
    
    # ----------------------- CONVOLUTION2D -----------------------
    '''
        --> Computes a 2-D convolution given 4-D input and filter tensors.
        --> Given an input tensor of shape [batch, in_height, in_width, in_channels] and 
            a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels], 
            this op performs the following:
        --> Flattens the filter to a 2-D matrix with shape [filter_height * filter_width * in_channels, output_channels].
        --> Extracts image patches from the input tensor to form a virtual tensor of shape [batch, out_height, out_width,
            filter_height * filter_width * in_channels].
        --> For each patch, right-multiplies the filter matrix and the image patch vector       
        
    '''
    conv2d = tf.nn.conv2d(input_x, W, strides = [1, 1, 1, 1], 
                          padding = 'SAME' # zero padding to the images
                         )
    
    
    # -------------------------- ACTUAL CONVOLUTION -------------------------- 
    '''
    Convoltuion is required for feature extraction purpose. We use variety of feature detectors(filters) 
    in this step to find a filter which extracts the most of the features from an image.
    
    '''
    
    convolved = tf.nn.relu(conv2d + b) # using 'relu' activation function
    
    
    # -------------------------- POOLING -------------------------- 
    '''
    Pooling is required mainly to deal with spatial variance in the images and also to reduce the size of image with
    the features preserved from image. This will help to avoid overfitting to an extent and 
    obviously reduces the computational intensity. Here, Max-pooling is used.
    
    ARGUMENTS:
    ----------
        value: A 4-D 'Tensor' with shape '[batch, height, width, channels]' and type 'tf.float32'.
        ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
        strides: A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.
        padding: A string, either 'VALID' or 'SAME'. 
        
    '''
    
    pooled = tf.nn.max_pool(convolved, # tensor that needs to be max-pooled [batch, height, width, color channel]
                            ksize = [1, 2, 2, 1], # size of the window for each dimension if input tensor
                            # as we want to apply pooling only on height and width of the image, its [1, 2, 2, 1]
                            strides = [1, 2, 2, 1], # stride of a slidding window
                            padding = 'SAME')
    
    return pooled    


def fully_connected_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = Weight_Init.init_weights([input_size, size])
    b = Weight_Init.init_bias([size])
    
    return tf.matmul(input_layer, W) + b    


# Placeholders for our input data in order to creater a tensor graph
x = tf.placeholder(tf.float32, shape = [None, 784]) # None is reserved for the actual batch size
# 784 is the number of pixels in our input data
y = tf.placeholder(tf.float32, shape = [None, 10])
# since our image is one hot enoded, the output is 10 dimensional value(each representing digits from 0-9)
# Ex: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] --> 3
#   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] --> 8

x_image = tf.reshape(x, [-1, 28, 28, 1]) # reshaping the flattened out array to actual image size. 
# 28 X 28 is the actual image size we have in input
# 1 represents only one color channel(using only gray-scale values)

conv1 = convolution_layer(x_image, shape = [6, 6, 1, 32])
# 6 X 6 convolutional layer
# 1 is input channel(using only gray-scale color)
# 32 features(number of output channels)
conv2 = convolution_layer(conv1, shape = [6, 6, 32, 64])
# since we have 32 outputs in the previous step, this layer is going to have 32 input layers
# its common rule to double the output channel in each following convolutional layer, hence 64


# -------------------------- FLATTENING -------------------------- 
flattened = tf.reshape(conv2, [-1, 7*7*64])
# flattening out the multi-dimensional array to feed to actual artificial neural net(fully connected layer)
# the original dimension of the image is 28 X 28, and we applied two pooling operations of stride 2.
# Hence the dimension becomes (28/2 = 14/2 = 7) 7 X 7 

# -------------------------- FULLY CONNECTED LAYER -------------------------- 
l1_output = tf.nn.relu(fully_connected_layer(flattened, 1024))

# Adding dropout regularization technique to avaoid possible overfitting(only in hidden layer)
dropout_rate = tf.placeholder(tf.float32) 
# fraction of number of neurons that we want to disable to the total number of neurons
l2_dropout = tf.nn.dropout(l1_output, keep_prob = 1 - dropout_rate)

y_pred = fully_connected_layer(l2_dropout, 10) # 10 possible outputs
# since this is the output layer, we no longer use dropout mechanism here


# Loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_pred))
# 'softmax' activation is used because of multiclass-classification problem

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
# an algorithm to find optimal set of weights for NN by varying the weights during stichastic gradient descent process

# Training the model
train = optimizer.minimize(cross_entropy)

# Initializing the global variables
init = tf.global_variables_initializer()

# Executing the graph
steps = 5000
saver = tf.train.Saver() # to save the trained model for future reference
with tf.Session(config = tf.ConfigProto(log_device_placement = True)) as sess:
    sess.run(init)
    
    for i in range(steps):
        batch_x, batch_y = mnist.train.next_batch(50) # holding only a batch of data in memory to train the model
        sess.run(train, feed_dict = {x: batch_x, y: batch_y, dropout_rate: 0.1})
        # It is recommended to start with dropout_rate = 0.1 and then see if it solves the overfitting and increase the value 
        # of p accordingly. We need to be aware that very high value of p will cause underfitting(more neurons will be 
        # disabled and the NN will not be able to learn enough)
        
        # Printing out the accuracy in every 100 steps
        if i%100 == 0:
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(matches,tf.float32)) # casting as float and taking the mean
            # Tesing the model
            acc = sess.run(accuracy, feed_dict = {x: mnist.test.images, y: mnist.test.labels, dropout_rate: 0})
            #  no dropout required here as this is a testing phase
            
            print('Step - %d: Accuracy = %.2f \n'%(i, acc*100))            
            
    saver.save(sess,'Models/handwritten_digits.ckpt') # saving the trained model