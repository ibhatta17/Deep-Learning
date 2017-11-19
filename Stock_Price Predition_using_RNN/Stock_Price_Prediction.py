###################################################################
#------------------- Part 1. Data Preprocessing -------------------
###################################################################

# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings

# Importing the Keras libraries and packages
from keras.models import Sequential # to initialize the neural network
from keras.layers import Dense # add hidden layers to the neural net
from keras.layers import LSTM # to add LSTM layers
from keras.layers import Dropout # for dropout regularization purpose

# regressor wrapper
from keras.wrappers.scikit_learn import KerasRegressor
# supressing the warnings
warnings.filterwarnings('ignore')

# --------------- Importing the training set --------------- 
dataset_train = pd.read_csv('Stock_Price_Train.csv')
# Since we are predicting only the Open stock price, we only need 'Open' columns
training_set = dataset_train.iloc[:, 1:2].values # Creates an array of one column with just 'Open' column

'''
# ---------------  Feature Scaling --------------- 
Using Normalization technique for feature scaling
As long as we are using the sigmoid function at the output layer of ANN, its recommended to use normalised scaler
instead of standard scaler
'''

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) # This will make sure all the stock price will lie between [0,1]
training_set = sc.fit_transform(training_set)

'''
--------------- Creating the data structure required to feed into RNN --------------- 
We need to identify the best number of timesteps(how far back we are looking into the RNN to update current timestamp weights)
Very high number of timesteps may cause overfitting and also the computional intensity
Creating a data structure with 60 timesteps and 1 output. Hence the next day price is predicted based on previous 6o days price
'''

X_train = [] # placeholder for i/p the of the RNN(stock price from last 60 days)
y_train = [] # placeholder for o/p of the RNN(stock price of next day)
tstep = 60
# since we are looking 60 timesteps back, we can start start looping over only after 60th record in our training set
for i in range(tstep, len(training_set)):
    X_train.append(training_set[i-tstep:i, 0]) # from indices 0 to 59
    y_train.append(training_set[i, 0]) # index 60
    
# converting the list to an array
X_train, y_train = np.array(X_train), np.array(y_train)

'''
--------------- Reshaping the data(adding more dimensionality to the training data) ---------------
Reference: https://keras.io/layers/recurrent/
Input shape
3D tensor with shape (batch_size, timesteps, input_dim).
'''

n_rows = X_train.shape[0]
n_cols = X_train.shape[1]
X_train = np.reshape(X_train, (n_rows, n_cols, 1)) # converting 2D array to 3D

###############################################################################
#------------------- Part 2. Building and Compiling the RNN -------------------
###############################################################################

def tunned_regressor(optimizer): 
    # Initializing the CNN
    regressor = Sequential()

    '''
	--------------- LSTM Layer ---------------
    We need LSTM layer(s) in RNN to avoid vanishing gradient problem.
    Vanishing gradient problem from Wikipedia
    In machine learning, the vanishing gradient problem is a difficulty found in training artificial neural 
    networks with gradient-based learning methods and backpropagation. In such methods, each of the neural network's
    weights receives an update proportional to the gradient of the error function with respect to the current weight 
    in each iteration of training. Traditional activation functions such as the hyperbolic tangent function have
    gradients in the range (−1, 1), and backpropagation computes gradients by the chain rule. 
    This has the effect of multiplying n of these small numbers to compute gradients of the "front" layers in 
    an n-layer network, meaning that the gradient (error signal) decreases exponentially with n while the front
    layers train very slowly
	'''
	
    # Adding the 1st LSTM layer
    regressor.add(LSTM(units = 50,# number of units(number of LSTM cells or memory units)
                       # Its a complex task to find the trends in the stock price we nedd a high number of neurons in our model
                       # a small number of units will not be sufficinet to cath the stock trend
                       return_sequences = True, # 'True' because we are creating a stacked RNN. So we are going to another LSTM
                       # We need to set to 'False' when we are not adding anymore LSTM layers
                       input_shape = (n_cols, 1) # here only the two last dimensions because the first one
                       # is the number of observation 
                       # and hence taken into consideration automatically
        )
    )
    # adding dropout regularization to avoid overfitting
    regressor.add(Dropout(0.2 # dropout rate
                          # 20% of the neurons will be dropped out during each training ite  ration
        ))

    # Adding 2nd LSTM layer and Dropout Regularization
    regressor.add(LSTM(units = 50, return_sequences = True, 
                       # input_shape parameters is required to set only in the first LSTM layer
        ))
    regressor.add(Dropout(0.2))

    # Adding 3rd LSTM layer and Dropout Regularization
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    # Adding 4th LSTM layer and Dropout Regularization
    regressor.add(LSTM(units = 50,
                       return_sequences = False # 'False' as we are not adding anymore LSTM layers
                      ))
    regressor.add(Dropout(0.2))

    # Adding the output layer
    regressor.add(Dense(units = 1 # '1' because we are predicting just 1 output variable
        ))

    regressor.compile(optimizer = optimizer, 
                       loss = 'mean_squared_error', # loss function for regression problem
                       )
    
    return regressor

##############################################################################
#------------------ Part 3. Training and Validating the RNN ------------------ 
##############################################################################

regressor = KerasRegressor(build_fn = tunned_regressor)

# using grid_search method to find the best possible values of hyper-parameters like batch_size and number of epocs etc
from sklearn.model_selection import GridSearchCV

parameters = {'batch_size': [32, 64], # whether to update the weights after each observation or after a batch of observation
              'epochs': [50, 100], # defines number of iterations
              'optimizer': ['adam', 'rmsprop'] # algorithm to find optimal set of weights for NN  
}

grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error', # mean squared error as performance measuring criteria
                           cv = 5                           
)

grid_search = grid_search.fit(X_train, y_train)

best_regressor = grid_search.best_estimator_ 

####################################################################
#------------------ Part 4. Making the Prediction ------------------ 
####################################################################

# Getting the actual stock price
dataset_test = pd.read_csv('Stock_Price_Test.csv')
actual_stock_price = dataset_test.iloc[:,1:2].values
'''
Predicting the stock price using the trainned RNN with best hyperparameters

Since we used the 60 previous stock price to predict the new stock price. 
Hence we need to have the same number of previous observations in order to make new prediction
Therefore we need to get some of those information from the training set given the fact that
we have only 20 days worth of data in our test set. So we will extract another 40 days worth of data from training set
'''

# we are concerned only about the 'Open' stock price
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), # concatenating test set and train set
                          axis = 0 # concatenating alog the vertical axis
                         ) 
# we only need 60 days worth of data. i.e. first day in test set - 60 days to last day in test set
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1) # to get proper array

# Since the inputs so far is directly from the original dataset and our RNN is trained in the normalized(scaled) dataset, 
# lets scale the inputs to have similar format as training set
inputs = sc.transform(inputs) # only transform methid instead of fit_transform because the sc object is already fitted

# Creating a data structure with 60 timesteps and 1 output for test set just like we did for training set
X_test = [] # placeholder for i/p the of the RNN(stock price from last 60 days)
for i in range(tstep, tstep + len(dataset_test)): # 60 to 80
    X_test.append(inputs[i-tstep:i, 0])
X_test = np.array(X_test)

# 2D to 3D -> adding dimension to convert the array to a format required by RNN
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) 

# using the trainned RNN to predict new stock price
predicted_stock_price = best_regressor.predict(X_test)

# Since the regressor was trained on scaled values, the predicted output is the scaled verson
# So we need to inverse_scale the outputs to get the stock price in original price range
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#####################################################################
#------------------ Part 5. Visualizing the Result ------------------ 
#####################################################################

plt.plot(actual_stock_price, color = 'red', label = 'Actual Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.savefig('Actual_vs_Predicted_Stock_Price.jpeg',format='jpeg')
plt.show()

############################################################
#------------------ 6. Evaluating the RNN ------------------ 
############################################################

from math import sqrt
from sklearn.metrics import mean_squared_error
rmse = sqrt(mean_squared_error(actual_stock_price, predicted_stock_price))
print('Root Mean Squared Error: ', rmse)