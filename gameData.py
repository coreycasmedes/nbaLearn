from nba_py import game
from sklearn.feature_extraction import DictVectorizer



boxscore = game.Boxscore("0041700404")
stats = boxscore.team_stats()


#print(stats)

#print(stats)

#use vector containing the nba data as input in sklearn learning
vec = DictVectorizer()

team_1_stats_dict = stats[0]
team_2_stats_dict = stats[1]
print("\n")
print(team_1_stats_dict)
print("\n")
print("\n")
team_1_stats_dict.pop('GAME_ID')
team_1_stats_dict.pop('TEAM_ID')
team_1_stats_dict.pop('TEAM_NAME')
team_1_stats_dict.pop('TEAM_ABBREVIATION')
team_1_stats_dict.pop('TEAM_CITY')
team_1_stats_dict.pop('MIN')
print(team_1_stats_dict)

team_2_stats_dict.pop('GAME_ID')
team_2_stats_dict.pop('TEAM_ID')
team_2_stats_dict.pop('TEAM_NAME')
team_2_stats_dict.pop('TEAM_ABBREVIATION')
team_2_stats_dict.pop('TEAM_CITY')
team_2_stats_dict.pop('MIN')

print("\n")
print("\n")
print(team_2_stats_dict)

#print(team_1_stats_dict)
#print("\n")
#print(team_2_stats_dict)

array = vec.fit_transform([team_1_stats_dict, team_2_stats_dict]).toarray()

#vec1 = vec.fit([team_1_stats_dict])
print(array[0])
#vec1 = vec.fit(team_1_stats)
#print(team_1_stats)
#print(team_1_stats)
#array = vec1[0]
print("\n")


#print(vec1)
#print(array[0])
#print(array[24])
#print(vec.get_feature_names())


#data = json.loads(part1)
#for x in range (len(stats)):


#print(team_1_stats['TEAM_NAME'] + " " + str(team_1_stats['PTS']))
#print(team_2_stats['TEAM_NAME'] + " " + str(team_2_stats['PTS']))
#print(stats.GameID)
#print(boxscore.team_stats())