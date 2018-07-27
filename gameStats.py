#credit to https://github.com/seemethere/nba_py for their python nba stats API utilized in this code.
from nba_py import game
from sklearn.feature_extraction import DictVectorizer
import numpy as np

#getGameStatsArray returns the boxscore stats for both teams in a 1D array and removes (1)independent features
#and (2)features that have string values to prevent one-hot-encoding, hence the features array will not be sparse.
#@param gameIDToken is the game ID string from stats.nba.com (example: 2016 NBA Finals Game 1: https://stats.nba.com/game/0041500401/  )
#@return sample an array containing to game stats for the 2 respective teams
def getGameStatsArray(gameIDToken):

	#using the python client nba_py for gathering statistics
	boxscore = game.Boxscore(gameIDToken)
	#you can find the values returned by Boxscore in the nba_py documentation or editing this code to use the stats object's property: get_feature_names 
	stats = boxscore.team_stats()

	team_1_stats_dict = stats[0]
	team_2_stats_dict = stats[1]
	#remove features from the python stats dictionary 
	team_1_stats_dict.pop('GAME_ID')
	team_1_stats_dict.pop('TEAM_ID')
	team_1_stats_dict.pop('TEAM_NAME')
	team_1_stats_dict.pop('TEAM_ABBREVIATION')
	team_1_stats_dict.pop('TEAM_CITY')
	team_1_stats_dict.pop('MIN')
	team_2_stats_dict.pop('GAME_ID')
	team_2_stats_dict.pop('TEAM_ID')
	team_2_stats_dict.pop('TEAM_NAME')
	team_2_stats_dict.pop('TEAM_ABBREVIATION')
	team_2_stats_dict.pop('TEAM_CITY')
	team_2_stats_dict.pop('MIN')

	#transform the python dictionary->array using DictVectorzier from sklearn
	vec = DictVectorizer()
	array = vec.fit_transform([team_1_stats_dict, team_2_stats_dict]).toarray()

	sample = np.append(array[0], array[1])

	return sample
