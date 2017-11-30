import numpy as np
import torch

class Convert_Data():    
    '''
    Creating a matrix with all the information we have, where row represents a user, column represent a movie and cell 
    represents a rating. Hence the dimensions of the matrix becomes number_of_users X number_of_movies
    0 in the matrix cell represents the user has not rated the moview yet.
    
    '''
    
    def __init__(self, training, test):
        self.training = training
        self.test = test
        self._matrix_size()
    
    def _matrix_size(self):
        # Let us first find the total number of users and total number of movies
        self.nb_users = int(max(max(self.training[:, 0]), max(self.test[:, 0]))) # first column of the training/test data
        # max of training/test as we are not sure which set holds the maximum value(the train/test split is random)
        self.nb_movies = int(max(max(self.training[:, 1]), max(self.test[:, 1]))) # 2nd column of the training/test data
      
    def _create_matrix(self, data):        
        new_data = []
        # Creating each set of matrix(list of lists) for training and set set
        # Both the test matrix and training matrix will be of same size
        for id_users in range(1, self.nb_users + 1): # list for each users. user_id starts from 1 in dataset
            id_movies = data[:, 1][data[:, 0] == id_users]  # movie_id rated by user 'id_users'
            # data[:, 1] retrieves the entire 'movie' column from the dataset. i.e [1, 2, 3, 4, 1, 2, 4] from above picture
            # [data[:, 0] == id_users] retrieves only those cells from the 'movie' column 
            # where the userid equals the current 'id_users' in the loop. i.e. only for a given user at a time
            # [1,2,3,4] in the 1st iteration and [1,2,4] in the 2nd iteration in the above example
            id_ratings = data[:, 2][data[:, 0] == id_users]
            # 0 in the matrix cell represents the user has not rated the movie yet.
            # this returns [5,2,1,3] in the 1st iteration and [2,1,3] in 3nd iteration
            ratings = np.zeros(self.nb_movies) # creating a list of all 0s
            ratings[id_movies - 1] = id_ratings # replacing 0 with a actual rating when user rates a movie
            # id_movies starts at 1. Hence need to make sure the index for 'ratings' starts at 0
            # returns[5,2,1,3] in the 1st iteration and [2,1,0,3] in the 2nd iteration
            new_data.append(list(ratings)) # creating list of lists
            
        return new_data        
        
    def convert(self):
        training = self._create_matrix(self.training)
        test = self._create_matrix(self.test)
        
        # Converting the data into Torch tensors
        training_set = torch.FloatTensor(training) # 'FloatTensor' expects list of lists
        test_set = torch.FloatTensor(test)
        
        return training_set, test_set    