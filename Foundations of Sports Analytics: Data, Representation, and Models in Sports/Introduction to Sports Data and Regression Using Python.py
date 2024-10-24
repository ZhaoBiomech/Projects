# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 21:17:56 2024

@author: William Zhao
"""
#%% Introduction to Regression Analysis using NHL Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import NHL Team Stats data
NHL_Team_Stats = pd.read_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\Data\\Files\\NHL_Team_Stats.csv')
NHL_Team_R_Stats = pd.read_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\Data\\Files\\NHL_Team_R_Stats.csv')

#To run regressions we will introduce a new library 'statsmodel'
#This provides classes and functions for the estimation of many different statistical models
import statsmodels.formula.api as sm

#We can use the ols() command to indicate an ordinary least squared regression
#The fit() function will allow us to obtain the estimated coefficient of our regression model
reg1 = sm.ols(formula = 'win_pct ~ goals_for' , data = NHL_Team_R_Stats).fit()
print(reg1.summary()) # We use the summary command to obtain the number of statistics from our reg model

#Now lets explore the relationship betwee goals_against and wpc in regular season
import seaborn as sns
sns.lmplot(x = 'goals_against' ,  y = 'win_pct' , data = NHL_Team_R_Stats)
plt.xlabel('Total Goals Against')
plt.ylabel('Win Percentage')
plt.title('Relationship between Goals and Winning Percentage' , fontsize = 20)

#Calculate the correlation coefficient (CC) between goals against and wpc
NHL_Team_R_Stats['goals_against'].corr(NHL_Team_R_Stats['win_pct']) #CC = -0.744

#Run a simple linear regression to find NHL team wpc as a function of goals against
reg2 = sm.ols(formula = 'win_pct ~ goals_against' , data = NHL_Team_R_Stats).fit()
print(reg2.summary())

#Often times, the outcome var of interest is affected by MULTIPLE FACTORS. 
#We can specify regression equations where the outcome is a function of more than one explanatory var
#Lets run a linear regression where winning percentage is a function of both average number of goals for and against per game
reg3 = sm.ols(formula = 'win_pct ~ avg_gf + avg_ga' , data = NHL_Team_R_Stats).fit()
print(reg3.summary())

#In the regressions above, we are using quantitative vars as explanatory vars
#But we can also use categorical vars in regressions as well
#To use categorical vars in a regression, we first transform it into a dummy var
#Lets consider the dataset that includes both regular season and playoff games
#First convert variable 'type' into categorical var
NHL_Team_Stats['type'] = NHL_Team_Stats['type'].astype(object)

#Now we can run a regression where wpc is a function of avg goals for and the type of competition
reg4 = sm.ols(formula = 'win_pct ~ avg_gf + type' , data = NHL_Team_Stats).fit() 
print(reg4.summary())
#Type 3  = playoff game and Type 2 = regular season game
#Interpretation:  with the same average goals for per game, the winning percentage 
#in the playoff games is 0.0160 (1.6%) lower than the winning percentage in the regular season games.

#What about interactions? We can run a regression that takes into account avg goals for, type of game, and the interaction between the two
reg5 = sm.ols(formula = 'win_pct ~ avg_gf + type + avg_gf * type' , data = NHL_Team_Stats).fit()
print(reg5.summary()) # For regular season games, scoring one more goal per game can increase wpc by 0.2365 (23.65%)
                      # For playoff games, scoring one more goal per game can increase wpc by 0.2365 - 0.0802 (%15.63)

#Self Test - Create the Pythagorean Expectation and then run a linear regression 
#Create our pyth var
NHL_Team_Stats['pyth_pct'] = NHL_Team_Stats['goals_for']**2 / (NHL_Team_Stats['goals_for']**2 + NHL_Team_Stats['goals_against']**2)

#Lets plot pyth vs wpc to examine our data
x = NHL_Team_Stats['win_pct']
y = NHL_Team_Stats['pyth_pct']
plt.scatter(x , y , s = 1 , c = 'r' , marker = '.' )

#Run a linear regression on the data
reg6 = sm.ols(formula = 'win_pct ~ pyth_pct' , data = NHL_Team_Stats).fit()
print(reg6.summary())

#Interpretation:
    #Rsquared of 0.779 means its generally a good fit
    #For every one unit increase in pyth_pct, win_pct will increase by 1.0673%

#Run a regression where wpc is the dv and pyth, competition, and interaction are ivs
reg7 = sm.ols(formula = 'win_pct ~ pyth_pct + competition_name + pyth_pct*competition_name' , data = NHL_Team_Stats).fit()
print(reg7.summary())

#%% Part 2- Regression Analyses with Cricket Data
#In this part we will use IPL player data to look at performance and its impact on salaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm  

#Import IPL player data
IPLPlayer = pd.read_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\Data\\Files\\IPL18Player.csv')
IPLPlayer.head() #Data exploration

IPLPlayer.info() # Checking for missing information
#There are missing values so we will drop observations with missing values
IPLPlayer = IPLPlayer.dropna()
IPLPlayer.shape

#First we will create variables that indicates the role of a player
IPLPlayer['batsman'] = np.where(IPLPlayer['matches_bowled'] > 0 , 1 , 0)
IPLPlayer['batsman'].describe()

IPLPlayer['bowler'] = np.where(IPLPlayer['matches_bowled'] > 0 , 1 , 0)

#We will now create performance measure variables that we can use in our regressions
#First we create a variable for the number of outs of a player
IPLPlayer['outs']= np.where(IPLPlayer['batsman'] == 1 , IPLPlayer['innings'] - IPLPlayer['not_outs'] , 0)

#Now we create batting avg, batting strike rate, bowling avg, and bowling strike rate
#We also add 1 to the number of outs, balls faced, and wickets taken in calcing this vars
IPLPlayer['batting_avg'] = IPLPlayer['runs'] / (IPLPlayer['outs'] + 1)
IPLPlayer['batting_strike'] = IPLPlayer['runs'] / ((IPLPlayer['balls_faced'] + 1)*100)
IPLPlayer['bowling_average'] = IPLPlayer['runs_conceded'] / (IPLPlayer['wickets']+1)
IPLPlayer['bowling_strike'] = IPLPlayer['balls_bowled'] / (IPLPlayer['wickets'] + 1)

#First lets run a regression of the salary on the type of player, batsman, bowler, and all-rounder
reg_IPL1 = sm.ols(formula = 'Salary ~ batsman + bowler + batsman * bowler' , data = IPLPlayer , missing = 'drop').fit()
print(reg_IPL1.summary()) #Not much correlation, shows that salary does not depend much on what your position is

#Instead, lets run some regressions on performance to see how it relates to salary.
#We will start with batsman
reg_IPL2 = sm.ols(formula = 'Salary ~ runs' , data = IPLPlayer).fit()
print(reg_IPL2.summary()) #R^2 of 0.267,, runs coeff = 1737.93,, for every run salary increases by $1738

reg_IPL3 = sm.ols(formula = 'Salary ~ runs + not_outs' , data = IPLPlayer).fit()
print(reg_IPL3.summary()) #R^2 of 0.318, runs coeff = 1491, not_outs coeff = 89550

reg_IPL4 = sm.ols(formula = 'Salary ~ runs + not_outs + balls_faced' , data = IPLPlayer).fit()
print(reg_IPL4.summary())

#In the next regressions, we will use the modified batting avg and battrike strike vars to measure player performance
reg_IPL5 = sm.ols(formula = 'Salary ~ batting_avg' , data = IPLPlayer) .fit()
print(reg_IPL5.summary()) # R^2 = 0.18 , batting_avg coeff = 1440

reg_IPL6 = sm.ols(formula = 'Salary ~ batting_avg + batting_strike' , data = IPLPlayer).fit()
print(reg_IPL6.summary()) # R^2 = 0.214 , batting_avg coeff = 1131 , batting_strike coeff = 2.5^e7

# Now we will turn to bowlers' perforamnce
reg_IPL7 = sm.ols(formula = 'Salary ~ runs_conceded' , data = IPLPlayer).fit()
print(reg_IPL7.summary()) # R^2 = 0.023 , runs_conceded coeff = 569

reg_IPL8 = sm.ols(formula = 'Salary ~ runs_conceded + balls_bowled' , data = IPLPlayer).fit()
print(reg_IPL8.summary()) # R^2 = 0.042 , runs_conceded coeff = -2749 , balls_bowled coeff = 4574

reg_IPL9 = sm.ols(formula = 'Salary ~ runs_conceded + balls_bowled + wickets' , data = IPLPlayer).fit()
print(reg_IPL9.summary()) # R^2 = 0.049

#In the next regression we will use the modified bowling avg and bowling strike vars to measure player performance
reg_IPL10= sm.ols(formula = 'Salary ~ bowling_average + bowling_strike' , data = IPLPlayer).fit()
print(reg_IPL10.summary()) #R^2 = 0.054

#Lastly we will incorporate performance measures of both batsman and bowler in same regression
reg_IPL11 = sm.ols(formula = 'Salary ~ runs + not_outs + balls_faced + runs_conceded + balls_bowled + wickets' , data = IPLPlayer).fit()
print(reg_IPL11.summary()) # R^2 = 0.408

reg_IPL12 = sm.ols(formula = 'Salary ~ batting_avg + batting_strike + bowling_average + bowling_strike' , data = IPLPlayer).fit()
print(reg_IPL12.summary())# R^2 = 0.324

#Run a final regression of salary as a function of the interaction of batsman and runs and the interaction of bowler and wickets taken
reg_IPL13 = sm.ols(formula = 'Salary ~ batsman*runs + bowler*wickets' , data = IPLPlayer).fit()
print(reg_IPL13.summary())

#%% Week 4 Assignment - Exploring and Running Regressions on NFL Data
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import datetime

#Import and Explore Data
#1)  Import the “nfl_game.csv” data file and name the dataframe as “NFL_Game” in Jupyter Notebook.
NFL = pd.read_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\Files\\nfl_game.csv')
#Descriptions of selected variables
#Unit of measurement of weather variables: 
    #temperature - degree Fahrenheit; 
    #humidity – relative percentage;
    #wind – mph
    #The variable “score” is the score earned by the team specified in the “team” variable. The variable “opponent_score” is the score earned by the team specified in the “opponent” variable.
    #The variable “score_diff” is defined as “score – opponent_score” for the given team.
    #The variable “stadium_age” is defined as the difference between the year of the season and the year the stadium opened.
    #The variable “stadium_neutral” indicates if the game was played in a third stadium, which is neither the home team’s nor the away team’s own stadium.

#2)Use the “describe” function to calculate summary statistics for the “date” variable. 
#Use the “describe” function to calculate summary statistics for the “score” variable based on whether it is a home or an away game for the team.
NFL['date'] = pd.to_datetime(NFL['date'])
NFL['date'].describe()

NFL.groupby(['home'])['score'].describe() #home = 1 , away = 0

#3)Find the correlation efficients between the following pairs of variables:
#“win” and “home”
reg_NFL1 = sm.ols(formula = 'win ~ home' , data = NFL).fit()
print(reg_NFL1.summary())       
#“score_diff” and “home”
reg_NFL2 = sm.ols(formula = 'score_diff ~ home' , data = NFL).fit()
print(reg_NFL2.summary())
#“score” and “weather_temperature”
reg_NFL3 = sm.ols(formula = 'score ~ weather_temperature' , data = NFL).fit()
print(reg_NFL3.summary())
#“score” and “weather_humidity”       
reg_NFL4 = sm.ols(formula = 'score ~ weather_humidity' , data = NFL).fit()
print(reg_NFL4.summary())
#“score” and “weather_wind_mph”
reg_NFL5 = sm.ols(formula = 'score ~ weather_wind_mph' , data = NFL).fit()
print(reg_NFL5.summary())

#Regression Analysis 1 – Test of Home Game Advantage 
#In this regression analysis, you will try to determine if playing at home gives teams any advantage in their performance.
#Run the following regressions where the variable “score_diff” is the dependent variable.
#Reg1_1: Include a single variable “home” as the independent variable.
reg_NFL6 = sm.ols(formula = 'score_diff ~ home' , data = NFL).fit()
print(reg_NFL6.summary())
#Reg1_2: Include the “home,” “stadium_capacity,” “stadium_neutral” variables and the interaction between “home” and “stadium_neutral” as the independent variables.
reg_NFL7 = sm.ols(formula = 'score_diff ~ home + stadium_capacity + stadium_neutral + home * stadium_neutral' , data = NFL).fit()
print(reg_NFL7.summary())
#Reg1_3: Include the “home”, “stadium_capcity,” and “stadium_neutral” variables, the interaction between “home” and “stadium_neutral,” as well as “team” and “opponent” as the independent variables.
reg_NFL8 = sm.ols(formula = 'score_diff ~ home + stadium_capacity + stadium_neutral + home * stadium_neutral + team + opponent' , data = NFL).fit()
print(reg_NFL8.summary())

#Regression Analysis 2 – Impact of Outside Factors on Scores
#In this regression analysis, you will investigate if the final score earned by each team is affected by outside factors such as the size and condition of the stadium as well as weather conditions at the stadium.
#Run the following regressions where the variable “score” is the dependent variable.
#Reg2_1: Include “season” and “home” variables as independent variables.
reg_NFL9 = sm.ols(formula = 'score ~ season + home' , data = NFL).fit()
print(reg_NFL9.summary())
#Reg2_2: Include “season,” “home,” “weather_temperature,” “weather_wind_mph,” and “weather_humidity” as independent variables.
reg_NFL10 = sm.ols(formula = 'score ~ season + home + weather_temperature + weather_wind_mph + weather_humidity' , data = NFL).fit()
print(reg_NFL10.summary())
#Reg2_3: Include “season,” “home,” “weather_temperature,” “weather_wind_mph,” “weather_humidity,” “stadium_capacity,” “stadium_age”, “stadium_type,” “stadium_neutral”, and the interaction between “home” and “stadium_neutral” as independent variables.     
reg_NFL11 = sm.ols(formula = 'score ~ season + home + weather_temperature + weather_wind_mph + weather_humidity + stadium_capacity + stadium_age + stadium_type + stadium_neutral + home * stadium_neutral' , data = NFL).fit()
print(reg_NFL11.summary())
#Reg2_4: Include “season,” “home,” “weather_temperature,” “weather_wind_mph,” “weather_humidity,” “stadium_capacity,” “stadium_age”, “stadium_type,” “stadium_neutral”, and the interaction between “home” and “stadium_neutral” as independent variables. Additionally, add both “team” and “opponent” in the regression.
reg_NFL12 = sm.ols(formula = 'score ~ season + home + weather_temperature + weather_wind_mph + weather_humidity + stadium_capacity + stadium_age + stadium_type + home*stadium_neutral + team + opponent' , data = NFL).fit()
print(reg_NFL12.summary())





