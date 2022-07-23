import os, random
import _pickle as pickle

import numpy as np
import pandas as pd

"""
Load the data and merge
"""
rating = pd.read_csv('./data/train/rating.txt', dtype=int, sep=' ',header=None)
bigtag = pd.read_csv('./data/train/bigtag.txt',dtype=int, sep=' ',header=None)
choicetag = pd.read_csv('./data/train/choicetag.txt',dtype=int, sep = ' ', header=None)
moviedata = pd.read_csv('./data/train/movie.txt',dtype=int, sep = ' ', header=None)

valid_data = pd.DataFrame(np.loadtxt('./data/valid/validation.txt',dtype=int))


rating.columns = ["userid", "movieid", "rating"]
bigtag.columns = ["userid", "movieid", "tagid"]
choicetag.columns = ["userid", "movieid", "tagid"]
moviedata.columns=["movieid", "tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7", "tag8"]

valid_data.columns = ["userid","tagid","label"]




"""
The auxiliary rating data
"""
tagList = ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7","tag8"]

# moviedata['tagList'] = moviedata[ ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7","tag8"] ]

rating  = pd.merge(rating, moviedata, how='left', on=['movieid'])
rating['movieAveRate'] = rating['rating'].groupby(rating['movieid']).transform('mean')

featureList = ["userid","tagid","movieid","rating","movieAveRate","tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7","tag8"]
movieTagDt =[]
for tagIndex in tagList:
    subColumns = featureList.copy()
    subColumns[1] = tagIndex
    subDt = rating[subColumns]
    subDt.columns = featureList
    movieTagDt.append(subDt)
movieTagDt =  pd.concat(movieTagDt)

"""
Create the training data for matrix factorization
"""
bigtag['mark'] = 'big'
choicetag['mark'] = 'choice'
totalTag = pd.concat([bigtag, choicetag])

totalTag = pd.merge(totalTag, moviedata, how='left', on=['movieid'])
totalTag['label'] = 1
totalTag.loc[totalTag.tagid==-1,'label'] = 0
tagList = ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7","tag8"]

"""
we split the dataset into 3 parts:
    (1) big, like
    (2) -1 dislike all
    (3) choice, like, dislike
"""
featureList = ["userid", "tagid","label"]
totalTagDt1 = totalTag.loc[(totalTag.mark =='big') & (totalTag.tagid != -1), featureList]
totalTagDt2 = totalTag.loc[totalTag.tagid == -1, ]
totalTagDt3 = totalTag.loc[(totalTag.mark=='choice') & (totalTag.tagid != -1),]

#The negative -1 data
trainDt = []
for tagIndex in tagList:
    subColumns = featureList.copy()
    subColumns[1] = tagIndex
    subDt = totalTagDt2[subColumns]
    subDt.columns = featureList
    trainDt.append(subDt)
totalTagDt2 =  pd.concat(trainDt)

# The choice tag data
useridList = totalTagDt3['userid'].unique()
trainDt2 = []
for subUserid in useridList:
   subDt = totalTagDt3.loc[totalTagDt3.userid==subUserid,]
   subMovieidList = subDt['movieid'].unique()
   for subMovieid in subMovieidList:
        subMovieTagList = moviedata.loc[moviedata.movieid==subMovieid,tagList].values[0]
        positiveTagList = subDt.loc[subDt.movieid==subMovieid,'tagid'].values
        negativeTagList = subMovieTagList[~np.isin(subMovieTagList, positiveTagList)]
        if(len(positiveTagList)>0):
            subsubDt1 = pd.DataFrame({'userid': subUserid, 'tagid': positiveTagList, 'label': 1})
            trainDt2.append(subsubDt1)
        if(len(negativeTagList)):
            subsubDt2 = pd.DataFrame({'userid': subUserid, 'tagid': negativeTagList, 'label': 0})
            trainDt2.append(subsubDt2)
totalTagDt3 = pd.concat(trainDt2)

totalDt = pd.concat([totalTagDt1, totalTagDt2, totalTagDt3])
totalDt.drop_duplicates(inplace=True)

"""
The data for R model
"""
totalDt = pd.merge(totalDt, movieTagDt, on=["userid","tagid"], how='inner')
# totalDt =totalDt.merge(totalDt, moviedata, on=["movieid"], how='left')
validDt = pd.merge(valid_data, movieTagDt, on=["userid","tagid"], how='inner')
# testDt = pd.merge(testDt, movieTagDt, on=["userid","tagid"], how='inner')

#ips
user_num = 1000
item_num = 1720
movie_num = 1000
#ut_propensity_score
totalDt_np = totalDt.to_numpy().astype(int)
validDt_np = validDt.to_numpy().astype(int)

p_y_count = np.bincount(totalDt_np[:, 2], minlength=2)[:]
p_y_o = p_y_count/ p_y_count.sum()

p_0 = p_y_count.sum() / (user_num * movie_num)

P_L_T = np.bincount(validDt_np[:, 2], minlength=2)[:]
p_y = P_L_T / P_L_T.sum()

propensity_score = p_y_o * p_0 / p_y

propensity_score = np.reciprocal(propensity_score)

totalDt['propensity_score_tag']  = np.where( totalDt['label'] == 0, propensity_score[0], propensity_score[1])

ddd = totalDt.to_numpy()

np.save('./data/train/totalDt',totalDt.to_numpy())

# np.savetxt("./data/train/taotalDt_columns.txt",aaa,fmt="%s")

# np.savetxt("./data/train/totalDt.txt",np.array(totalDt),fmt="%d")