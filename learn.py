import gameStats
import numpy as np
from sklearn import linear_model

print("**the test: [0] for cavs win, [1] for warriors win")

features = [gameStats.getGameStatsArray("0041500401"), gameStats.getGameStatsArray("0041500402"), gameStats.getGameStatsArray("0041500403"), gameStats.getGameStatsArray("0041500404"), gameStats.getGameStatsArray("0041500405"), gameStats.getGameStatsArray("0041500406")]

# 0 for cavs win,	1 for warriors win
#the labels subspace should be a subspace of the labels provided subspace 
labels = [1, 1, 0, 1, 0, 0]

#predict game 7 of the 2016 NBA finals
X = [gameStats.getGameStatsArray("0041500407")]


#create our logisitc regression model 
clf = linear_model.LogisticRegression()
clf.fit(features, labels)

game7prediction = clf.predict(X)

print(game7prediction)

