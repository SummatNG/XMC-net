import os, random
import _pickle as pickle

import numpy as np
import pandas as pd


data_rats = pd.read_csv('./data/Movielens/ml-latest-small/ratings.csv')
data_movs = pd.read_csv('./data/Movielens/ml-latest-small/movies.csv')
data_tags = pd.read_csv('./data/Movielens/ml-latest-small/tags.csv')


rat_mov = pd.merge(data_rats, data_movs, how='left', on=['movieId'])



i=1