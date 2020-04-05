# importing the libraries

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import os

# -------------- Paths -----------
BASE = os.getcwd()
DATASET_PATH_MOVIE = BASE + '/Dataset/ml-1m/movies.dat'
DATASET_PATH_USERS = BASE + '/Dataset/ml-1m/users.dat'
DATASET_PATH_RATING = BASE + '/Dataset/ml-1m/ratings.dat'
TRAINING_SET_DIR = BASE + '/Dataset/ml-100k/u1.base'
TEST_SET_DIR = BASE + '/Dataset/ml-100k/u1.test'

# importing the dataset

movies = pd.read_csv(DATASET_PATH_MOVIE
                     , sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv(DATASET_PATH_USERS, sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv(DATASET_PATH_RATING, sep='::', header=None, engine='python', encoding='latin-1')

# train test split 80-20
training_set = pd.read_csv(TRAINING_SET_DIR, delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv(TEST_SET_DIR, delimiter='\t')
test_set = np.array(test_set, dtype='int')

# getting the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))


# converting the data into and array with users in lines and movies in column
def convert(data):
    new_data = []
    for id_user in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_user]
        id_ratings = data[:, 2][data[:, 0] == id_user]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data


training_set = convert(training_set)
test_set = convert(test_set)

# converting the data into torch tensor
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)