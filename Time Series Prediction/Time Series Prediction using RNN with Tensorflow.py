# **  Predicting the milk production using historical data**
# ** Source: https://datamarket.com/data/set/22ox/monthly-milk-production-pounds-per-cow-jan-62-dec-75#!ds=22ox&display=line **
# **Monthly milk production: pounds per cow Jan '62 - Dec '75 **

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Reading the data
data = pd.read_csv('monthly-milk-production.csv', index_col = 'Month')
data.index = pd.to_datetime(data.index) # convering the date to 'DATETEIEM' format

# Visualizing the time series data
data.plot()
plt.show()

#len(data)

# Train Test Split
# Instead of a random train-test split, we want to use the sequential data for both test and train set.
train_set = data.head(156)
# will predict for a year
test_set = data.tail(12)

# Scaling the data
sc = MinMaxScaler()
train_data = sc.fit_transform(train_set)
test_data = sc.transform(test_set)

# Batch function to feed a batch of data at a time

def next_batch(training_data, batch_size, steps):
    '''
    INPUT: Data, Batch Size, Time Steps per batch
    OUTPUT: A tuple of y time series results. y[:,:-1] and y[:,1:]
    
    '''
    
    # Geeting random starting point for each batch.
    # Using np.random.randint to set a random starting point index for the batch.
    # Each batch needs have the same number of steps in it. i.e limiting the starting point to len(data) - steps
    rand_start = np.random.randint(0, len(training_data) - steps)
    
    # Making sure the random initialization lies in actual time series
    # the random start to random start + steps + 1. Then reshape this data to be (1, steps + 1)
    y_batch = np.array(training_data[rand_start: rand_start + steps + 1]).reshape(1, steps + 1)
    
    # Returning two batches to return y[:,:-1] and y[:,1:]. 
    # First series with actual batch and 2nd series with time shifted batch
    return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)

# ---------------------- RNN Model ----------------------

num_inputs = 1 # Just one feature, the time series
num_time_steps = 12 # Num of steps in each batch
num_neurons = 100 # 100 neuron
num_outputs = 1 # Just one output, predicted time series
batch_size = 1 # Size of the batch of data

# Placeholders for inpout and output
x = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

# Creating RNN Cell
# Also play around with GRUCell
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicLSTMCell(num_units = num_neurons, activation=tf.nn.relu),
    output_size=num_outputs
) 

outputs, states = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y)) # Mean squared error Loss function
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001) # Adam optimizer to optimize gradient descent
model = optimizer.minimize(loss) # training the model to minimize the loss function

# Executing the tensor graph
init = tf.global_variables_initializer()
saver = tf.train.Saver() # to save the trained model
gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9) # to avoid memory freeze

with tf.Session(config = tf.ConfigProto(gpu_options = gpu_option)) as sess:
    sess.run(init)
    
    iterations = 6000 # how many iterations to go through (training epochs)
    for i in range(iterations):
        x_batch, y_batch = next_batch(train_data, batch_size, num_time_steps) # training each batch at a time
        sess.run(model, feed_dict = {x: x_batch, y:y_batch})
    
        # Prining the loss in each 100 itirations
        if i%100 == 0:
            mse = loss.eval(feed_dict = {x: x_batch, y:y_batch})
            print('Step %d \t MSE %.5f'%(i, mse))
            
    saver.save(sess, 'Models/time_series') # saving the trained model

with tf.Session() as sess:    
    # Restoring the saved trained model
    saver.restore(sess, "Models/time_series")

    # Numpy array for last 12 month frm training set for future prediction
    # training set data. Hint: Just use tail(12) and then pass it to an np.array
    train_seed = list(train_data[-12:])
    
    # Looping thru all the months 
    for iteration in range(12):
        x_batch = np.array(train_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={x: x_batch})
        train_seed.append(y_pred[0, -1, 0])

# the model was built with scaled input data. now, we need to inverse the scaling to get actual values
result = sc.inverse_transform(np.array(train_seed[12:])).reshape(12, 1)
# we append the predicted data on the 'train_seed'. The first 12 items are the actual train values and 
# the remaining are the predicted ones. Hence sclicing thru [12:]  to get only the preducted values

# adding the result to test dataframe to compare
test_set['Predicted'] = result

# Plotting the actual vs predicted values
test_set.plot()
plt.show()