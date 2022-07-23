import numpy as np
import pandas as pd
import random


bigtag = np.loadtxt('./data/train/bigtag.txt',dtype=int)
choicetag = np.loadtxt('./data/train/choicetag.txt',dtype=int)
movie_data = np.loadtxt('./data/train/movie.txt',dtype=int)
rating = np.loadtxt('./data/train/rating.txt',dtype=int)

movie = []
for i in range(movie_data.shape[0]):
    tmp = movie_data[i,1:]
    movie.append(tmp)
tag_num = np.max(movie)



mat = np.zeros((1000,tag_num+1)) # [man, tag_num+1]
all_data_array = []
bigtag_array = []
choicetag_array = []

for i in range(bigtag.shape[0]):
    if bigtag[i][2] != -1:
        mat[bigtag[i][0]][bigtag[i][2]] = 1
        all_data_array.append([bigtag[i][0],bigtag[i][2],1])
        bigtag_array.append([bigtag[i][0],bigtag[i][2],1])
    if bigtag[i][2] == -1:
        for tag in movie[bigtag[i][1]]:
            mat[bigtag[i][0]][tag] = -1
            all_data_array.append([bigtag[i][0],tag,0])
            bigtag_array.append([bigtag[i][0],tag,0])

# # extract deterministic data from choicetag
for i in range(choicetag.shape[0]):
    if choicetag[i][2] != -1:
        mat[choicetag[i][0]][choicetag[i][2]] = 1
        all_data_array.append([choicetag[i][0],choicetag[i][2],1])
        choicetag_array.append([choicetag[i][0],choicetag[i][2],1])
    if choicetag[i][2] == -1:
        for tag in movie[choicetag[i][1]]:
            mat[choicetag[i][0]][tag] = -1
            all_data_array.append([choicetag[i][0],tag,0])
            choicetag_array.append([choicetag[i][0],tag,0])
for i in range(choicetag.shape[0]):
    if choicetag[i][2] != -1:
        for tag in movie[choicetag[i][1]]:
            if mat[choicetag[i][0]][tag] == 0:
                mat[choicetag[i][0]][tag] = -1
                all_data_array.append([choicetag[i][0],tag,0])
                choicetag_array.append([choicetag[i][0],tag,0])


# Unique
all_data_array = np.array(all_data_array)
print(all_data_array.shape[0])
print(np.count_nonzero(all_data_array[:,2]))
all_data_array = [tuple(row) for row in all_data_array]
all_data_array = np.unique(all_data_array, axis=0)
print(all_data_array.shape[0])
print(np.count_nonzero(all_data_array[:,2]))


# Unique
bigtag_array = np.array(bigtag_array)
print(bigtag_array.shape[0])
print(np.count_nonzero(bigtag_array[:,2]))
bigtag_array = [tuple(row) for row in bigtag_array]
bigtag_array = np.unique(bigtag_array, axis=0)
print(bigtag_array.shape[0])
print(np.count_nonzero(bigtag_array[:,2]))


# Unique
choicetag_array = np.array(choicetag_array)
print(choicetag_array.shape[0])
print(np.count_nonzero(choicetag_array[:,2]))
choicetag_array = [tuple(row) for row in choicetag_array]
choicetag_array = np.unique(choicetag_array, axis=0)
print(choicetag_array.shape[0])
print(np.count_nonzero(choicetag_array[:,2]))


np.savetxt("./baseline_data/extract_bigtag.txt",np.array(bigtag_array),fmt="%d")
np.savetxt("./baseline_data/extract_choicetag.txt",np.array(choicetag_array),fmt="%d")
np.savetxt("./baseline_data/extract_alldata.txt",np.array(all_data_array),fmt="%d")
