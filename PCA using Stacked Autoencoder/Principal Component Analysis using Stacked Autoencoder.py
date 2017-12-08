# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Reading the data
df = pd.read_csv('anonymized_data.csv')
df.shape

# Scaling the data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
data = sc.fit_transform(df.drop('Label', axis = 1)) 
# Dropping the 'Label' column as this is a unsupervised method for dimensionality reduction

# 30 --> 15 --> 2 dimensionality reduction
no_of_features = 30
no_of_hidden_layers1 = 15 
no_of_hidden_layers2 = 2 
no_of_hidden_layers3 = no_of_hidden_layers1 # 30 --> 15 --> 2 --> 15 --> 30
no_of_output = no_of_features # this is the requirement for Autoencoder 
# as we will try to regenarate the original input from teh reduced-dimension data

initializer = tf.variance_scaling_initializer()
# the weights between the layers are quite different
# variance_scaling allows to adpat with the difference in weights between different hidden layers of the autoencoder
# its capable of adapting its scale to the shape of the weights tensors
# a lot better than random initializing same weights across the board

# Weights
w1 = tf.Variable(initializer([no_of_features, no_of_hidden_layers1]), dtype=tf.float32)
w2 = tf.Variable(initializer([no_of_hidden_layers1, no_of_hidden_layers2]), dtype=tf.float32)
w3 = tf.Variable(initializer([no_of_hidden_layers2, no_of_hidden_layers3]), dtype=tf.float32)
w4 = tf.Variable(initializer([no_of_hidden_layers3, no_of_output]), dtype=tf.float32)

# Biases 
b1 = tf.Variable(tf.zeros(no_of_hidden_layers1))
b2 = tf.Variable(tf.zeros(no_of_hidden_layers2))
b3 = tf.Variable(tf.zeros(no_of_hidden_layers3))
b4 = tf.Variable(tf.zeros(no_of_output))

# Placeholder for input data
x = tf.placeholder(tf.float32, shape= [None, no_of_features])
# Since autoencoder is aan unsupervised algorithm, we only need to create a placeholder form input features only.

# Defining the stacked layers
hid_layer1 = tf.nn.relu(tf.matmul(x, w1) + b1) # relu activation function
hid_layer2 = tf.nn.relu(tf.matmul(hid_layer1, w2) + b2)
hid_layer3 = tf.nn.relu(tf.matmul(hid_layer2, w3) + b3)
output = tf.nn.relu(tf.matmul(hid_layer3, w4) + b4)

# defining Mean Squared Error as loss function
loss = tf.reduce_mean(tf.square(output - x)) # as we are trying to regenerate the output from the reduced dimensions

# Using Adam optimizer for optimizing stochastic grdadient descent process
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)

# Trainign the model to reduce the loss
model = optimizer.minimize(loss)

# Executing the graph
init = tf.global_variables_initializer()
iterations = 500

with tf.Session() as sess:
    sess.run(init)
    for _ in range(iterations):
        sess.run(model, feed_dict = {x: data})
        
    # Evaluating hidden layer 2 output(the reduced dimension data(2D data in our case))
    reduced_dim_data = hid_layer2.eval(feed_dict = {x: data})  
    # we will use this reduced dimension data for further applications down the data pipeline

# Plotting the 2D data
plt.scatter(reduced_dim_data[:, 0], reduced_dim_data[:, 1], c = df['Label'])
plt.show()