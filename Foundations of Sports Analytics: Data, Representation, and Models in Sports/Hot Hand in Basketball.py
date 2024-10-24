# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:45:58 2024

@author: William Zhao
"""
#%% Part 1 - Understanding and Cleaning the NBA Shot Log Data
#We will use the 2016-2017 NBA shot log data to demonstrate how to test the hot hand

import numpy as np
import pandas as pd
import datetime as dt

#Import NBA shot data
NBA_shot = pd.read_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\Shotlog_16_17.csv') 
NBA_shot.head()
NBA_shot.info()

#First let's create some useful vars 
NBA_shot['current_shot_hit'] = np.where(NBA_shot['current_shot_outcome'] == 'SCORED' , 1 , 0) #dummy variavle to indicate the success of a shot

NBA_shot['date'] = pd.to_datetime(NBA_shot['date']) #convert date to a datetime variable

NBA_shot['time'] = pd.to_timedelta('00:' + NBA_shot['time']) #convert time to a date time variable by first adding '00:' to add the hour
NBA_shot['time'].describe() 

#We will also create a lag variable for shot to indicate the result of the previous shot by
#the same player in the same game
NBA_shot['lag_shot_hit'] = NBA_shot.sort_values(by = ['quarter' , 'time'] , ascending = [True , False]).groupby(['shoot_player' , 'date'])['current_shot_hit'].shift(1)

#We can now sort the df by player, game(date) , quarter, and time of shot
NBA_shot.sort_values(by = ['shoot_player' , 'date' , 'quarter' , 'time'] , ascending = [True , True, True, True])

#Now lets create a df for average success of players over a season
#Since the 'current_shot_hit' var is a dummy var , the avg of this var would indicate success rate 
Player_stats = NBA_shot.groupby(['shoot_player'])['current_shot_hit'].mean() #Group by each player and average out their shot success rate
Player_stats = Player_stats.reset_index()
Player_stats.rename(columns = {'current_shot_hit' : 'average_hit'} , inplace = True)

#We will use the players statistics to analyze the hot hand, so we will merge player stats back into the NBA df
NBA_shot = pd.merge(NBA_shot , Player_stats , on = ['shoot_player'])

#Create a var to indicate the total number of shots in the dataset for each player
Player_Shots = NBA_shot.groupby(['shoot_player']).size().reset_index(name ='shot_count')
Player_Shots.sort_values(by = ['shot_count'] , ascending = False)

#We also need to consider that players have diff number of shots in each individual game
#We will need to treat the data differently for a player who had only two shots in a game 
#compared to those who had attempted 30 in a game.
#Create a var to indicate the number of shots in each game by player
Player_Game = NBA_shot.groupby(['shoot_player' , 'date']).size().reset_index(name = 'shot_per_game')
Player_Game

#We will now merge these two new vars back into the original df
NBA_shot = pd.merge(NBA_shot , Player_Shots , on = ['shoot_player'])
NBA_shot = pd.merge(NBA_shot , Player_Game , on = ['shoot_player' , 'date'])

#Lets sort the data again
NBA_shot.sort_values(by = ['shoot_player' , 'date' , 'quarter' , 'time'] , ascending = [True , True , True , True])

#Finally, we will treat the 'points' and 'quarter' vars as objects
NBA_shot['points'] = NBA_shot['points'].astype(object)
NBA_shot['quarter'] = NBA_shot['quarter'].astype(object)

#And we will drop all missing observations in the lagged variable
NBA_shot = NBA_shot[pd.notnull(NBA_shot['lag_shot_hit'])]

#We will save our data
NBA_shot.to_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\Shotlog1.csv' , index = False)
Player_Shots.to_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\Player_shots1.csv' , index = False)
Player_Game.to_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\Player_game1.csv' , index = False)
Player_stats.to_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\Player_stats1.csv' , index = False)

#%% Using Summary Statistics to Examine the 'Hot Hand'

import pandas as pd
import numpy as np
import datetime as dt

Shotlog = pd.read_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\Shotlog1.csv')
Player_Stats = pd.read_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\Player_stats1.csv')
Player_Shots = pd.read_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\Player_shots1.csv')
Player_Game = pd.read_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\Player_game1.csv')

#We can first calculate the conditional probability of making a shot in the current
#period conditional on making the previous shot
#Conditional Probability = Probability of Making Consecutive Shots / Probability of Making Previous Shot

#First we will need to create a var that indicates a player made consecutive shots
Shotlog['conse_shot_hit'] = np.where((Shotlog['current_shot_hit'] == 1) & (Shotlog['lag_shot_hit'] == 1) , 1 , 0)
Shotlog.head()

#Then we create a player-level df. The avg of conse_shot_hit would be the joint probability 
#of making current and previous shots. We will also calc the avg of lag_shot_hit to indicate
#the probability of making the previous shot
Player_Prob = Shotlog.groupby(['shoot_player'])[['conse_shot_hit' , 'lag_shot_hit']].mean()
Player_Prob.reset_index()
Player_Prob.rename(columns  = {'lag_shot_hit' : 'average_lag_hit'} , inplace = True)

#Now we can calculate the conditional probability for each player
#We can calc the conditional probability by dividing the joint probability by
#the probability of making the previous shot
Player_Prob['conditional_prob'] = Player_Prob['conse_shot_hit'] / Player_Prob['average_lag_hit']

#Merge the probability with the other player stats
Player_Stats = pd.merge(Player_Stats , Player_Prob , on = ['shoot_player'])

#Note when we created the conditional probability var, some observations may have been missing
#so we will drop these observations
Player_Stats = Player_Stats[pd.notnull(Player_Stats['conditional_prob'])]

#Let's sort the data so we can see who is most likely to have hot hand
Player_Stats.sort_values(by = ['conditional_prob'] , ascending = False).head(10)

#We can also compare cond prob with avg hit. First we create a var for the difference
#between conditional and unconditional probabilities 
Player_Stats['diff_prob'] = Player_Stats['conditional_prob'] - Player_Stats['average_hit']
Player_Stats = pd.merge(Player_Stats , Player_Shots , on = ['shoot_player'])
Player_Stats.sort_values(by = ['diff_prob'] , ascending = [False]).head(10)

#T-test for statistical significance on the difference
#We can use a t-test to test if the player's probabilities of hitting
#the goal is statistically significantly different than their conditional prob
import scipy.stats as sp

sp.stats.ttest_ind(Player_Stats['conditional_prob'] , Player_Stats['average_hit'])
#p-value of 0.10 which is > 0.05 so not statistically significant 

#We can calc the autocorrrelation coefficient by calcing the correlation coefficient
#between the 'current_shot_hit' and the 'lag_shot_hit' var
Shotlog['current_shot_hit'].corr(Shotlog['lag_shot_hit']) # ACC = -0.0066

#Rather than calculate the ACC for the population, we should calculate it
#for every player
Shotlog.groupby('shoot_player')[['current_shot_hit' , 'lag_shot_hit']].corr().head(10)

#We may not want to print out a 2x2 matrix for every player. We can use the unstack() command to reshape the data
Autocorr_Hit = Shotlog.groupby('shoot_player')[['current_shot_hit' , 'lag_shot_hit']].corr().unstack()
Autocorr_Hit.head() #Note that now each row represents a single player, but we still have duplicate info in the columns

#We can use the .iloc command to select the columns we need
#In the iloc[] command, we specify the rows we want, then the columns [rows , columns]
#To select all rows and the 2nd column -> [:,1]
Autocorr_Hit = Shotlog.groupby('shoot_player')[['current_shot_hit' , 'lag_shot_hit']].corr().unstack().iloc[:,1].reset_index()

#Notice that we still have 2 levels of var names
#We can use the 'get_level_values' command to reset the var name to the first level (index 0)
Autocorr_Hit.columns = Autocorr_Hit.columns.get_level_values(0)

#Now we rename the var to capture the auto correlation coefficient
Autocorr_Hit.rename(columns = {'current_shot_hit' : 'autocorr'} , inplace = True)

#How informative the ACC also depends on the number of shots per game for each player
#Lets add the number of shots per game to the ACC matrix and then sort by size of ACC
Player_Game_Shot = Player_Game.groupby(['shoot_player'])['shot_per_game'].mean().reset_index(name = 'avg_shot_game')

Autocorr_Hit = pd.merge(Autocorr_Hit , Player_Game_Shot , on = ['shoot_player'])
Autocorr_Hit.sort_values(by = ['autocorr'] , ascending = [False]).head(10)

#We will now merge the Player_Game_Shot df to the Player_Shots df since both
#are measured in player level and both contain info on number of shots
Player_Shots = pd.merge(Player_Shots , Player_Game_Shot , on = ['shoot_player'])

#Save Data
Shotlog.to_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\Shotlog2.csv' , index=False)
Player_Stats.to_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\Player_Stats2.csv' , index=False)
Player_Shots.to_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\Player_Shots2.csv', index=False)

#%% Using Regression Analysis to Test the 'Hot Hand'
#In this section we will use regression analysis to test for the 'hot hand'

import pandas as pd
import numpy as np
import datetime as dt
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

Shotlog = pd.read_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\Shotlog2.csv')
Player_Stats = pd.read_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\Player_Stats2.csv')
Player_Shots = pd.read_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\Player_Shots2.csv')

#Prediction Error
#Lets create a var that equals the diff between the outcome of the shot and the avg success rate
#Since we use avg success rate to predict outcome, this difference will capture the prediciton error
Shotlog['error'] = Shotlog['current_shot_hit'] - Shotlog['average_hit']
Shotlog['lagerror'] = Shotlog['lag_shot_hit']- Shotlog['average_hit']

#We can graph the outcome of the shots to see if there is any pattern over time in the var
#We will look at LeBron James' performance during the reg season as an example
Shotlog['time'] = pd.to_timedelta(Shotlog['time'])
Shotlog['time'].describe()

#We will first graph the outcome of LBJ's shots in a single game on April 9th 2017
#To do this we will use a small trick. We ask Python to produce a line plot
#with a line width of 0 , because scatter plots require the x-axis to be numeric, and not a date or time var
Shotlog[(Shotlog.shoot_player == 'Lebron James') & (Shotlog.date == '2017-04-09')].plot(x = 'time' , y = 'current_shot_hit' , marker = 'o' , linewidth = 0)
#This won't work

#First we will subset a dataset that includes only LBJ's data
LeBron_James = Shotlog[(Shotlog.shoot_player == 'LeBron James')]

#Now we can graph prediction error for LBJ for all games separately in season
g = sns.FacetGrid(LeBron_James, col="date", col_wrap=4)
g = g.map(plt.plot, "time", "current_shot_hit", marker='o', linewidth=0)
g.set_axis_labels("Game", "Shots");

#We will do a similar exercise for the statistics for Cheick Diallo
Cheick_Diallo=Shotlog[(Shotlog.shoot_player == 'Cheick Diallo')]
g = sns.FacetGrid(Cheick_Diallo, col="date", col_wrap=4)
g = g.map(plt.plot, "time", "current_shot_hit", marker='o', linewidth=0)

#Regression analysis on prediction error
#We will first run a simple regression of the prediction error of current period
#on the prediction error of previous period
reg1 = sm.ols(formula = 'error ~ lagerror' , data = Shotlog).fit()
print(reg1.summary()) #R^2 = 0 meaning not a good fit at all!

#There are alot of factors that may influece the success of shot, for example,
#the player's own skill as a shooter, atmosphere, and whether its the beginning or end of game
#Lets add these control vars in our regression
reg2 = sm.ols(formula = 'error ~ lagerror + player_position + home_game + opponent_previous_shot + C(points) + time_from_last_shot + C(quarter)' , data = Shotlog).fit()
print(reg2.summary())
#Note that the R^2 = 0.015 which is still very small

#As we have seen, some players had many shots per game, while some had just a few
#Differnet players may have different variations in their success rate in the shots.
#We can run a weighted least squared regression to address this problem
#Weighted least squares estimation weights the observations proportional to the
#reciprocal of the error variance of the observation. Thus weighted least sqaures
#can overcome the issue of non-constant variance

#We can use the sm.wls command to run the weighted least square regression
#weighted by the number of shot per game (weight = 1/shot_per_game)
reg3 = sm.wls(formula = 'error ~ lagerror + player_position + home_game + opponent_previous_shot + points + time_from_last_shot + quarter' , weights = 1/Shotlog['shot_per_game'] , data = Shotlog).fit()
print(reg3.summary())

#Regression analysis on individual players
#Run a regression of current error on lagged error for LeBron James
reg_LeBron = sm.ols(formula = 'error ~ lagerror + home_game + opponent_previous_shot + C(points) + time_from_last_shot + C(quarter)' , data = LeBron_James).fit()
print(reg_LeBron.summary())

reg_LeBron_wls = sm.wls(formula = 'error ~ lagerror + home_game + opponent_previous_shot + points + time_from_last_shot + quarter' , weights = 1/LeBron_James['shot_per_game'] , data = LeBron_James).fit()
print(reg_LeBron_wls.summary())

#We can also take a look back at LeBron James ACC
Shotlog[(Shotlog.shoot_player == 'LeBron James')][['current_shot_hit' , 'lag_shot_hit']].corr()

#The ACC between the outcomes of the current shot and the previous shot for LeBron James is very small
#We can do a similar exercise for Cheick Diallo. 
Cheick_Diallo=Shotlog[(Shotlog.shoot_player == 'Cheick Diallo')]

reg_Diallo = sm.ols(formula = 'error ~ lagerror + home_game + opponent_previous_shot + C(points) + time_from_last_shot + C(quarter)' , data = Cheick_Diallo).fit()
print(reg_Diallo.summary())

reg_Diallo_wls= sm.wls(formula = 'error ~ lagerror + home_game + opponent_previous_shot + points + time_from_last_shot + quarter' , weight = 1/Cheick_Diallo['shot_per_game'] , data = Cheick_Diallo).fit()
print(reg_Diallo_wls.summary())

#Instead of running regressions manually for each player, we can 
#define a function to run regressions for each player

#Define a function to run a OLS regression by player

def reg_player(player):
    Shotlog_player = Shotlog[Shotlog.shoot_player == player]
    reg_player = sm.ols(formula = 'error ~ lagerror + home_game + opponent_previous_shot + points + time_from_last_shot + quarter' , data = Shotlog_player).fit()
    print(reg_player.summary())
    return;

#We can now use this function for individual players, for example Russell Westbrook
reg_player('Russell Westbrook')
reg_player('Dwight Howard')

#We can also refine a function to run a WLS regression by player
def reg_wls_player(player):
    Shotlog_player = Shotlog[Shotlog.shoot_player == player]
    reg_wls_player = sm.wls(formula = 'error ~ lagerror + home_game + opponent_previous_shot + points + time_from_last_shot + quarter' , weights = 1 / Shotlog_player['shot_per_game'], data = Shotlog_player).fit()
    print(reg_wls_player.summary())
    return;

reg_wls_player('Russell Westbrook')

#*We can extract estimated coefficient on the lagged error for each player
#First we will examine how to do it manually for one player, then we
#will builda  loop to do it for all players

#First create a list of unique player names
player_list = np.array(Shotlog['shoot_player']) #Create an array with all names of players
player_list = np.unique(player_list) #Make each entry unique 
player_list[0]

#Run a weighted least sqaures regression for each player by specifiying
# shoot_player == player_list[index]
Shotlog_player = Shotlog[Shotlog.shoot_player == player_list[0]]
reg_player = sm.wls(formula = 'error ~ lagerror + home_game + opponent_previous_shot + points + time_from_last_shot + quarter' , weights = 1 / Shotlog_player['shot_per_game'] , data = Shotlog_player).fit()
print(reg_player.summary())

#Extract the estimated coefficients, along with the p-value and t-statistics
#of the estimates and store them in a dataframe
RegParams = pd.DataFrame(reg_player.params).reset_index()
RegTvals = pd.DataFrame(reg_player.tvalues).reset_index()
RegPvals = pd.DataFrame(reg_player.pvalues).reset_index()

RegOutput = pd.merge(RegParams , RegTvals , on = ['index'])
RegOutput = pd.merge (RegOutput , RegPvals , on = ['index'])
RegOutput

#Finally, we can write a loop to extract regression outputs for each player
i = 0 #initialize a variable 'i' and set it to 0. This var will be used as a counter or index within the loop
Player_Results = {} # Initialize an empty dictionary to store key values 
while i <= len(player_list) - 1: #Start a 'while' loop. A 'while' loop continues to execute as long as the condition specified remains TRUE
    Shotlog_player = Shotlog[Shotlog.shoot_player == player_list[i]]
    reg_player = sm.wls(formula = 'error ~ lagerror + home_game + opponent_previous_shot + points + time_from_last_shot + quarter' , weights = 1 / Shotlog_player['shot_per_game'] , data = Shotlog_player).fit()
    RegParams = pd.DataFrame(reg_player.params).reset_index()
    RegTvals = pd.DataFrame(reg_player.tvalues).reset_index()
    RegPvals = pd.DataFrame(reg_player.pvalues).reset_index()
    
    RegOutput = pd.merge(RegParams , RegTvals , on = ['index'])
    RegOuput = pd.merge (RegOutput , RegPvals , on = ['index'])
    
    LagErr = RegOuput[RegOutput['index'] == 'lagerror']
    LagErr = LagErr.drop(columns = ['index'])
    LagErr = LagErr.rename(columns = {'0_x' : 'Coef' , '0_y' : 'T_Statistics' , 0:'P_Value'})
    LagErr['shoot_player'] = player_list[i]
    Headers = ['shoot_player' , 'Coef' , 'T_Statistics' , 'P_Value']
    Player_Results[i] = LagErr[Headers]
    i = i+1
    
#Write another loop to build a df to store the regression output for all players
RegPlayer = Player_Results[0]
j = 1
while j <= len(player_list) - 1:
    RegPlayer = pd.concat([RegPlayer, Player_Results[j]], ignore_index=True)
    j = j+1
RegPlayer = RegPlayer.reset_index()
RegPlayer = RegPlayer.drop(columns=['index'])
RegPlayer

#Now merge the total number of shots captured in 'Player_Shots'
#to the regression result df. This total number represents the sample
#size of each regression
RegPlayer = pd.merge(RegPlayer , Player_Shots , on = ['shoot_player'])
RegPlayer.head()

#Finally, we can display players with statistically sig estimates on the lagged error var
display(RegPlayer.loc[RegPlayer['P_Value'] <= 0.05])

#Save the updated dfs
Shotlog.to_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\Shotlog3.csv' , index=False)
Player_Stats.to_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\Player_Stats3.csv' , index=False)
Player_Shots.to_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\Player_Shots3.csv', index=False)

#%% Assigment 6 - Examining 'Hot Hand' in NBA 2014 - 2015 Shot Data
#Part 1- Data Preparation and Exploration
import numpy as np
import pandas as pd
import datetime as dt
#Import the “Shotlog_14_15.csv” data file as “Shotlog_1415” 
#Import “Player_Stats_1415.csv” data file as “Player_Stats” 
Player_Stats = pd.read_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\Player_Stats_14_15.csv')
Shotlog_1415 = pd.read_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\Shotlog_14_15.csv')

#Descriptions of the datasets and selected variables
#a)In the dataset “Shotlog_14_15,” each observation represents an attempt of a shot. 
    #In the dataset “Player_Stats_14_15,” each observation represents a player.
#b)The “average_hit” variable in both dataframes indicate the average success rate 
    #of a player making a shot over the season. It is defined and calculated the same way in both dataframes.
#c)The variable “home_away” indicates whether the team that the player belongs to played at home or away.
#d)The variable “result” indicates whether the team that the player belongs to won 
    #or lost the game. The variable “final_margin” represents the difference in final score between the team the player belongs to and their opponent’s.
#e)The variable “shot_number” is the order of the shot the given player attempted at the given game.
#f)“game_clock” is the countdown clock for each quarter. The game clock starts at 
    #12 minutes. “shot_clock” refers to the display of a countdown clock of the time
    #within which the team possessing the ball must attempt a field goal. The shot clock starts at 24 seconds.

#Convert the “date” variable to a date type variable and calculate summary statistics for the “shot_clock” variable.
Shotlog_1415['date'] = pd.to_datetime(Shotlog_1415['date'])

#Create a lagged variable “lag_shot_hit” to indicate the result of the previous 
#shot by the same player at the same game.
#Hint: In this dataset, the variable “match” may not be able to uniquely identify 
#each game; you can use “game_id” instead. You can sort the data by shot number for 
#each player to create the lagged variable. 
Shotlog_1415['lag_shot_hit'] = Shotlog_1415.sort_values(by = ['shot_number' , 'shoot_player'] , ascending = [True , False]).groupby(['shoot_player' , 'date'])['current_shot_hit'].shift(1)

#Create a variable “error” to indicate the prediction error for each shot and a 
#variable “lagerror” for the prediction error for the previous shot. 
#The “error” variable is defined as the difference between the outcome of the current shot 
#and the average success rate (“average_hit”) and 
#the “lagerror” variable is defined as the difference between the outcome of the previous shot and the average success rate.
Shotlog_1415['error'] = Shotlog_1415['current_shot_hit'] - Shotlog_1415['average_hit']
Shotlog_1415['lagerror'] = Shotlog_1415['lag_shot_hit']- Shotlog_1415['average_hit']

#Calculate summary statistics for the “error” and “lagerror” variables.
Shotlog_1415['error'].describe()
Shotlog_1415['lagerror'].describe()

#%% Part 2 - Conditional Probability and Autocorrelation
import scipy.stats as sp
#1.Create a dummy variable “conse_shot” that indicates a player made consecutive shots. 
Shotlog_1415['conse_shot'] = np.where((Shotlog_1415['current_shot_hit'] == 1) & (Shotlog_1415['lag_shot_hit'] == 1) , 1 , 0)

#2.Create a dataframe “Player_Prob” for the probability of making the previous 
#shot and the joint probability for making both the previous and current shots. 
#Name the probability of making the previous shot “average_lag_hit” and the probability of making both shots “conse_shot_hit.” 
Player_Prob = Shotlog_1415.groupby(['shoot_player'])[['conse_shot' , 'lag_shot_hit']].mean()
Player_Prob.reset_index()
Player_Prob.rename(columns  = {'lag_shot_hit' : 'average_lag_hit' , 'conse_shot' : 'conse_shot_hit'} , inplace = True)

#3.In the “Player_Prob” dataframe, calculate the conditional probability “conditional_prob” 
#for a player to make a shot given that he made the previous shot. 
Player_Prob['conditional_prob'] = Player_Prob['conse_shot_hit'] / Player_Prob['average_lag_hit']

#4.Merge the “Player_Prob” dataframe into the “Player_Stats” dataframe.
Player_Stats = pd.merge(Player_Stats , Player_Prob , on = ['shoot_player'])

#5.Calculate summary statistics for the probability for a player to make a shot (“average_hit”) 
#and the conditional probability for a player to make a shot given that he made the previous one 
#(“conditional_prob”) and the probability of players making consecutive shots (“conse_shot_hit”).
Player_Stats['average_hit'].describe()
Player_Stats['conditional_prob'].describe()
Player_Stats['conse_shot_hit'].describe()

#6.Perform a t-test for the statistical significance on the difference between conditional 
#probability and unconditional probability of making a shot.
sp.stats.ttest_ind(Player_Stats['conditional_prob'] , Player_Stats['average_hit'])

#7.Calculate the first order autocorrelation coefficient on making a shot 
#(correlation coefficient between making the current shot and the previous shot) for the entire shotlog dataset.
Shotlog_1415['current_shot_hit'].corr(Shotlog_1415['lag_shot_hit']) # ACC = -0.0105

#8.Calculate the first order autocorrelation coefficient on making a shot for each player.
#Display the top ten players with the highest first order autocorrelation coefficient.
Autocorr_Hit = Shotlog_1415.groupby('shoot_player')[['current_shot_hit' , 'lag_shot_hit']].corr().unstack().iloc[:,1].reset_index()

Autocorr_Hit.columns = Autocorr_Hit.columns.get_level_values(0)

Autocorr_Hit.rename(columns = {'current_shot_hit' : 'autocorr'} , inplace = True)

Autocorr_Hit.sort_values(by = ['autocorr'] , ascending = [False]).head(10)

#%% Part 3 - Regression Analysis
import statsmodels.formula.api as sm
#In this section, you will run several regressions to investigate the “hot hand.” 
#In all the regressions, the dependent variable is “error” and the independent variable of interest is “lagerror.” 
Shotlog_1415.info()
Shotlog_1415['points'] = Shotlog_1415['points'].astype(object)
Shotlog_1415['quarter'] = Shotlog_1415['quarter'].astype(object)

#Reg1: Run a linear least squares regression using the entire shotlog dataframe. Include the following control variables:
#Shot distance, Number of dribbles,nTouch time, Type of shot (“points” variable), Quarter of the game (as a categorical variable), Home or away game, Shoot_player, Closest defender, Closest defender distance
reg1 = sm.ols(formula='error ~ lagerror + shot_dist + dribbles + touch_time + points + C(quarter) + home_away + shoot_player + closest_defender + closest_def_dist', data=Shotlog_1415).fit()
print(reg1.summary())

#Reg2: Run a weighted least squares regression using the entire shotlog dataframe. Include the same set of control variables as in Reg1. The regression should be weighted by the number of shot per game (weight=1/shot_per_game).
reg2= sm.wls(formula='error ~ lagerror + shot_dist + dribbles + touch_time + points + C(quarter) + home_away + shoot_player + closest_defender + closest_def_dist', weights = 1 / Shotlog_1415['shot_per_game'] , data=Shotlog_1415).fit()
print(reg2.summary())

#Reg3_player: Run linear least squares regressions on individual players. Include the following control variables:
#Shot distance, Number of dribbles, Touch time, Type of shot (“points” variable), Quarter of the game (as a categorical variable), Home or away game, Closest defender distance
player_list = np.array(Shotlog_1415['shoot_player']) #Create an array with all names of players
player_list = np.unique(player_list) #Make each entry unique 
Shotlog_player = Shotlog_1415[Shotlog_1415.shoot_player == player_list[0]]

def reg_player(player):
    Shotlog_player = Shotlog_1415[Shotlog_1415.shoot_player == player]
    reg_player = sm.ols(formula = 'error ~ lagerror + shot_dist + dribbles + touch_time + points + C(quarter) + home_away + closest_def_dist' , data = Shotlog_player).fit()
    print(reg_player.summary())
    return;

#We can now use this function for individual players, record the coeff for shot_dist
reg_player('russell westbrook') # -0.0148
reg_player('stephen curry') # -0.0181
reg_player('james harden') # -0.0154
reg_player('andrew wiggins') # -0.0212

#Reg4_wls_player: Run weighted least squares regressions on individual players. Include the same set of control variables as in Reg3. The regression should be weighted by the number of shot per game (weight=1/shot_per_game).
i = 0 
Player_Results = {} 
while i <= len(player_list) - 1: 
    Shotlog_player = Shotlog_1415[Shotlog_1415.shoot_player == player_list[i]]
    reg_player = sm.wls(formula = 'error ~ lagerror + shot_dist + dribbles + touch_time + points + C(quarter) + home_away + closest_def_dist' , weights = 1 / Shotlog_player['shot_per_game'] , data = Shotlog_player).fit()
    RegParams = pd.DataFrame(reg_player.params).reset_index()
    RegTvals = pd.DataFrame(reg_player.tvalues).reset_index()
    RegPvals = pd.DataFrame(reg_player.pvalues).reset_index()
    
    RegOutput = pd.merge(RegParams , RegTvals , on = ['index'])
    RegOuput = pd.merge (RegOutput , RegPvals , on = ['index'])
    
    LagErr = RegOuput[RegOutput['index'] == 'lagerror']
    LagErr = LagErr.drop(columns = ['index'])
    LagErr = LagErr.rename(columns = {'0_x' : 'Coef' , '0_y' : 'T_Statistics' , 0:'P_Value'})
    LagErr['shoot_player'] = player_list[i]
    Headers = ['shoot_player' , 'Coef' , 'T_Statistics' , 'P_Value']
    Player_Results[i] = LagErr[Headers]
    i = i+1
    
#Write another loop to build a df to store the regression output for all players
wlsRegPlayer = Player_Results[0]
j = 1
while j <= len(player_list) - 1:
    wlsRegPlayer = pd.concat([wlsRegPlayer, Player_Results[j]], ignore_index=True)
    j = j+1
wlsRegPlayer = wlsRegPlayer.reset_index()
wlsRegPlayer = wlsRegPlayer.drop(columns=['index'])
wlsRegPlayer

#Neither
#andrew wiggins
#reggie
















































