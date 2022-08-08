# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 06:50:05 2022

@author: Rakesh
"""
###############Below coding is being referred from Github accounts########
########################PROBLEM 1#####################################
import pandas as pd

##loading dataset#

game = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_Recommendation Engine/game.csv' , encoding= 'utf-8')

from sklearn.feature_extraction.text import TfidfVectorizer
#term frequency - inverse document frequency is numerical statistic that is intended#
#reflect how important a word is to document in a collection or corpus#
##creating Tfidfvectorizer to remove all stop words ##
tfidf = TfidfVectorizer(stop_words='english')

##lets check for NA and null value just in case#
game.isnull().sum()
game.isna().sum()

###preparing TFIDF matrix by fiting and transformation method#
tfidf_matrix = tfidf.fit_transform(game.game) #transform count to nirmalized TF#
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

##computing cosine similiarity on Tfidf matrix#
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

#creating a mapping  of game name to index no##
game_index = pd.Series(game.index,index= game['game'])
game_index = game_index[~game_index.index.duplicated(keep='first')]

game_id = game_index['Grand Theft Auto V']
game_id
#getting top 10##using it title#
def get_recommendations(Name , TopN):
    game_id = game_index[Name]
    
###getting pairwise similiarity score for all game with that game#
cosine_score = list(enumerate(cosine_sim_matrix[game_id]))   

##sorting cosine similiarity based on score##
cosine_score = sorted(cosine_score, key=lambda x:x[1], reverse=True) 
    
#getting score to TopN most similiar game#
cosine_score_N = cosine_score[0: 10+1]

##getting game index##
game_idx = [i[0] for i in cosine_score_N]
game_score = [i[1] for i in cosine_score_N ]

#similiar game and score##
game_similiar_show = pd.DataFrame(columns=['name', 'Score'])
game_similiar_show['name']= game.iloc[game_idx,1]
game_similiar_show['Score']= game_score
game_similiar_show.reset_index(inplace=True)
print(game_similiar_show)

##ERnter your game & num of game to get reccomendation #
get_recommendations('Batman: Arkham City', TopN = 10)
game_index['Batman: Arkham City']

#################################Problem 2###########################################
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
#loading dataset#
entertainment =  pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_Recommendation Engine/Entertainment.csv' , encoding= 'utf-8')

entertainment.columns
entertainment.shape
##creating Tfidfvectorizer to remove all stop words ##
tfidf = TfidfVectorizer(stop_words= 'english')

entertainment['Category'].isnull().sum()
###preparing TFIDF matrix by fiting and transformation method#
tfidf_matrix = tfidf.fit_transform(entertainment.Category)
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel
##computing cosine similiarity on Tfidf matrix#
cosine_sim_matrix= linear_kernel(tfidf_matrix,tfidf_matrix)
#creating a mapping  of movie name to index no##
entertainment_index = pd.Series(entertainment.index, index = entertainment['Titles']).drop_duplicates()

#lets checking index no#
entertainment_id= entertainment_index['Waiting to Exhale (1995)']
entertainment_id

def get_recommendations(Name,TopN):
    entertainment_id=entertainment_index[Name]

###getting pairwise similiarity score for all movie with that movie#
cosine_score = list(enumerate(cosine_sim_matrix[entertainment_id]))

##sorting cosine similiarity based on score##
cosine_score = sorted(cosine_score, key= lambda x:x[1], reverse = True)

#getting score to TopN most similiar movie#
cosine_score_N = cosine_score[0 : 10+1]
##getting movie index##
entertainment_idx = [i[0] for i in cosine_score_N]
entertainment_score = [i[1] for i in cosine_score_N]

#similiar movie and score##
entertainment_similiar_show = pd.DataFrame(columns=['name' , 'Score'])
entertainment_similiar_show['name'] = entertainment.loc[entertainment_idx,'Titles']
entertainment_similiar_show['Score'] = entertainment_score
entertainment_similiar_show.reset_index(inplace=True)
print(entertainment_similiar_show)
##Enter your movie & num of game to get reccomendation #
get_recommendations('Toy Story (1995)', TopN=10)
entertainment_index['Toy Story (1995)']
