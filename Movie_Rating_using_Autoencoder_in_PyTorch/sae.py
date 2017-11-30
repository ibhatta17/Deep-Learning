import torch.nn as nn

# Creating the architecture of Autoencoder
# using Stacked Autoencoder(SAE) here
class SAE(nn.Module): # Inheriting nn.Module
    def __init__(self, nb_movies):
        super(SAE, self).__init__() # to get all method and class from nn.Module
        # first full connection between input(all user ratings) and first hidden layer of SAE
        self.fc1 = nn.Linear(nb_movies, # number of features
                            20 # number of neurons in first hidden layer(tunable parameter)
                            )
        # second full connection(between 1st hidden layer(with 20 hidden layers) and 2nd hidden layer)
        self.fc2 = nn.Linear(20, 10) # 10 neorns in 2nd hidden layer. feature detection based on detected features from layer 1
        # 3rd full connection
        self.fc3 = nn.Linear(10, 20) # Decoding 2nd hidden layer
        # 4th full connection
        self.fc4 = nn.Linear(20, nb_movies) # Decoding 1st hidden layer
        self.activation = nn.Sigmoid() # sigmoid activation function to activate the neuron
        
    def forward(self, x): # for forward propagation of encoder and decoder
        '''
        Input:
        x: input vecotr of features(movie ratings)
        
        Output:
        x: vector of predicted movie ratings
        
        '''
        
        # encoding 1st hidden layer(nb_movies ---> 20)
        x = self.activation(self.fc1(x)) # encoed vector for 1st fully connected layer
        # encoding 2nd hidden layer(20 ---> 10)
        x = self.activation(self.fc2(x)) # encoed vector for 2nd fully connected layer
        # decoding 3rd hidden layer(10 ---> 20)
        x = self.activation(self.fc3(x)) # decoded vector for 3rd fully connected layer
        # decoding 4th hidden layer(20 ---> nb_movies)
        # since this is the output layer, we do not use activation fuction
        x = self.fc4(x) # decoded vector for 4th fully connected layer
        
        return x 