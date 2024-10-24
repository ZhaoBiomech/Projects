# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 13:33:03 2024

@author: William Zhao
"""
#%% Accessing data using Python
# We will use NBA data to demonstrate how to import data into Python
# How to clean up data before conducting analyses, as well as how to 
# describe and summarize the data

import pandas as pd
import numpy as np

#Import NBA team data
NBA_Teams=pd.read_csv('C:\\Users\William Zhao\\Desktop\\Data Science Learning\\Data\\nba_teams.csv')
NBA_Teams

NBA_Teams.shape #Shape function to show how many variables and observations in our dataset

#We can rename variables for ease of reading and understanding. The first variable is "unnamed" so lets change it to team number. 
NBA_Teams.rename(columns = {'Unnamed: 0' : 'Team_Number'},inplace = True) # Inplace 'True' parameter indicates to replace the old variable with a new name. 'Fa;se' would create a new variable with the new name
NBA_Teams.rename(columns = {'ID' : 'TEAM_ID'} , inplace = True)
NBA_Teams.rename(columns={'FULL_NAME' : 'TEAM_NAME'} , inplace = True)
NBA_Teams

#We can drop a variable, ie delete a column, we can use the 'drop' command
#The variable Team_Number has little meaning so we can drop it
NBA_Teams.drop(['Team_Number'] , axis = 1, inplace = True) #The argument 'axis=1' tells Python that we are dropping a column, not a row. To drop a row would be '= 0'

#Now we will work on our game level data.
#We start by importing the data
Games = pd.read_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\basketball_games.csv')
Games.head() # We can display just the first 5 rows of the dataset using the 'head' command

#From examining the head of the dataset, we notice that there are WNBA and NBA2K games in the dataset
#We want to drop these observations and only focus on NBA Games
Games.drop([0] , axis=0, inplace = True) # Drop the row (axis = 0) in index 0 

#However, a more efficient way to drop observations is based on certain conditions
Games = Games[Games.TEAM_NAME !="Las Vegas Aces"] #In this case we are not using the drop function, we can specify our TEAM_NAME variable to be not equal to 'Las Vegas Aces'
Games = Games[Games.TEAM_NAME !="Phoenix Mercury"]

#To only focus on NBA games we could merge the NBA_Teams and Games datasets to filter out NBA games
NBA_Games = pd.merge(NBA_Teams, Games, on = ['TEAM_ID' , 'TEAM_NAME']) # Merge the datasets on Team ID and Team Name. 

#To understand and clean the merged dataset, we can first look at all the variables
NBA_Games.columns #This command displays all the variable names

#The variable ABBREVIATION appears twice and carries the same information so we can get rid of it
NBA_Games.drop(['ABBREVIATION'] , axis = 1, inplace = True , errors = 'ignore')

#Currently the dataset is sorted by the criteria we used to merge the datasets (ie Team ID) but we can sort it by other criteria, such as date of game
NBA_Games.sort_values(by = ['GAME_ID'], ascending = [False]).head(20) #Sort the data set in descending order by Game ID and then display the top 20 observations

#Before we do any analysis, we need to check for missing values. 
#We can use the info command which will return the number of observations that have real numbers. Then we can compare to see if any variable is missing observations
NBA_Games.info()
NBA_Games.notnull() #We can also use the notnull function to detect where the missing values are
NBA_Games.isnull()

#To handle missing values we can do 2 things
# 1) We drop the observations with missing values
# 2) We replace the missing values with valid values (imputation), such as median or mean
NBA_Games = NBA_Games[pd.notnull(NBA_Games['FG_PCT'])] #Detect which rows have missing values in the column FG_PCT, and then replace the original dataset with only rows that contain values in that column
NBA_Games.shape

NBA_Game = NBA_Games.fillna(NBA_Games.mean())
NBA_Games.info()

#We can create variables equal to the total number of goals made
NBA_Games['GM'] = NBA_Games['FGM'] + NBA_Games['FG3M'] + NBA_Games['FTM'] #Goals made = field goals made + field goals 3pt made + free throws made

NBA_Games['GA'] = NBA_Games['FGA'] + NBA_Games['FG3A'] + NBA_Games['FTA'] #Goals attempted = field goals attempted + field goals 3pt attempted + free throws attempted

#We can also create variables based on the condition of the value of another variable (we will drop this variable since its not needed)
NBA_Games['RESULT'] = np.where(NBA_Games['PLUS_MINUS'] > 0, 'W' , 'L') #Create a result variable where W is assigned when Plus Minus is greater than 0
NBA_Games.drop(['RESULT'] , axis = 1, inplace = True)

#We can also create a variable based on the difference between two other variables
NBA_Games.sort_values(['GAME_ID' , 'WL'] , inplace=True) #First sort the values by game id and WL because each game has two rows representing both teams
NBA_Games['POINT_DIFF'] = NBA_Games.groupby(['GAME_ID'])['PTS'].diff() #Group by Game id and calculate the difference in points
#The POINT_DIFF variable only has the point difference for the winning team, we need to impute the difference for the losing team as well
NBA_Games['POINT_DIFF'] = NBA_Games['POINT_DIFF'].fillna(NBA_Games.groupby('GAME_ID')['POINT_DIFF'].transform('mean')) #Fill all the Nans in the POINT_DIFF column with the mean of the values in the group, grouped by GameID

#We can also drop all observations with missing value in at least one variable
NBA_Games = NBA_Games.dropna()# Drop all observations with missing values

#To work with season level data rather than team level data, we can create a new dataset that includes aggregate info of team statistics in each season
NBA_Team_Stats = NBA_Games.groupby(['TEAM_ID' , 'SEASON_ID'])[['PTS' , 'FGM' , 'FGA' , 'FG_PCT' , 'FG3M' , 'FG3A' , 'FG3_PCT' , 'FTM' , 'FTA' , 'FT_PCT' , 'OREB' , 'DREB' , 'REB' ,'AST' , 'STL' , 'TOV' , 'PF' , 'PLUS_MINUS']].sum()
NBA_Team_Stats

NBA_Team_Stats = NBA_Team_Stats.reset_index() #The newly created dataset had 2 levels of index, TeamID and SeasonID, so we reset the index

#Now we can create a df that equals the total number of observations within a specified group using the size command
NBA_Game_Count = NBA_Games.groupby(['TEAM_ID' , 'SEASON_ID']).size().reset_index(name='GAME_COUNT') # Create a df that equals the total number of games played by a team in each season by the size of each group

#Now we can save our data
NBA_Games.to_csv('C:\\Users\William Zhao\\Desktop\\Data Science Learning\\Data\\NBA_Games.csv' , index=False) #Save the merged data to the file path without adding the index as a column in the csv file

#%% Data Exploration and Summary Statistics

#Continuing on with our created NBA Games data set from the above module
import pandas as pd
NBA_Games = pd.read_csv('C:\\Users\William Zhao\\Desktop\\Data Science Learning\\Data\\NBA_Games.csv')

#To explore the dataset we can assess the variable types using the dtypes command
#In data analysis we often convert categorical variables (objects) into dummy variables
NBA_Games.dtypes #Notice that WL are objects 

#The variable WL only carries 2 values, win or lose. We will create dummy variables to capture the categories
dummy = pd.get_dummies(NBA_Games, columns = ['WL']) #convert a categorical variable to dummy variable. This function will also omit any missing value
dummy.columns #Notice that 2 variables were created, WL_L and WL_W. 

#We can attach the "WL_W" dummy variable back to our original dataset now
NBA_Games = pd.concat([NBA_Games, dummy['WL_W']], axis=1) #Concat just the specified column from dummy to NBA Games

#Rename the WL_W variable to just Win
NBA_Games.rename(columns = {'WL_W' : "WIN"} , inplace=True)

#In sports we often work with date and time data. The date variable is originally stored as an object
#which means each date is treated equally without ordering. We can use the
#pd.todatetime command to convert object variable to date variable
import datetime
NBA_Games['GAME_DATE'] = pd.to_datetime(NBA_Games['GAME_DATE']) #Convert Game Date to dtype: datetime64

#Descriptive and Summary Analyses
#We can use the describe() command to calculate summary statistics
#Returns basic summary stats for all numerical variables
NBA_Games.describe()

NBA_Games.describe(include='all') #We can also include non-numerical variables using the argument "include=all'

NBA_Games['PTS'].describe() #We can summarize a single variable as well

#We can also calculate individual statistics by using mean() , median() , std()
NBA_Games['FGM'].mean()
NBA_Games['FGM'].median()
NBA_Games['FGM'].std()

#We can also calculate summary statistics of a variable based on another variable
#usually based on a different categorical variable
NBA_Games.groupby(['WL'])['PTS'].mean() #Calculate the mean points for wins and losses
NBA_Games.groupby(['WL'])['REB'].mean()

#We can summarize the date variable to gather valuable info
NBA_Games['GAME_DATE'].describe() #50% is the date that appears the most, min and max are earliest and latest days

#Visualizing the Data
#We can visualize the distribution of a variable using a histogram
NBA_Games.hist(column = 'PTS')

NBA_Games.hist(column = 'PTS' , bins = 20) #We can specify the number of bins in a histogram

NBA_Games.hist(column = 'PTS' , bins = 20, rwidth = 0.9) #For visual appeal, we can add space between bins. We can narrow the bin width to 0.9

#Finally, we save the edited dataset
NBA_Games.to_csv('C:\\Users\William Zhao\\Desktop\\Data Science Learning\\Data\\NBA_Games2.csv' , index=False)

#%% Summary Statistics and Correlation Analysis
#Import updated NBA Games data
import pandas as pd
NBA_Games = pd.read_csv('C:\\Users\William Zhao\\Desktop\\Data Science Learning\\Data\\NBA_Games2.csv')

#Central Tendency vs Variation
#We will compare the success rates of 2-point goals and 3-point goals to demonstrate
#the difference between central tendency and variation
NBA_Games['FG_PCT'].describe() #Summary stats for 2-point goals success
NBA_Games['FG3_PCT'].describe() #Summary stats for 3-point goals success

#Avg success rate for 2FG is ~45% and 3FG is ~35%, median is 45.2% and 35% meaning
#Half the teams have success rates lower than those
#STD of 0.056 for 2PT and 0.099 means that there is greater variation in 3PT
#Now lets compare the distribution of 2PT and 3PT using histogram
NBA_Games.hist(column = ['FG_PCT' , 'FG3_PCT'] , bins = 20 , sharex=True , sharey = True) # sharex and sharey ask if we want to restrict the same range of x and y for the 2 histograms

#We will now introduce and use matplotlib which provides more useful functions to make plots
import matplotlib.pyplot as plt

NBA_Games[['FG_PCT' , 'FG3_PCT']].plot.hist(alpha = 0.3 , bins = 20) #alpha specifies transparency, so that the 2 histograms don't block each other. alpha= 0 : fully transparent, alpha = 1 : fully opaque 
plt.xlabel('Field Gold Percentage') #Add xaxis title
plt.ylabel('Frequency') #Add yaxis title
plt.title('Distribution of Field Goal Percentages' , fontsize = 15) #add title and specify font size
plt.savefig('FG_PCT_Distributions.png') #export and save the file as a png

#We can also create a histogram by the result of the game using the "by" option
NBA_Games.hist(by = "WL" , column = 'FG_PCT' , color = 'red' , bins = 15 , sharex = True , sharey = True)
plt.savefig('FG_PCT_WL.png')

#Lets create Time-Series graphs

#First change the datatype of "GAME_DATE" from object to datetime
import datetime
NBA_Games['GAME_DATE'] = pd.to_datetime(NBA_Games['GAME_DATE'])

#Subsetting a dataset: The dataset we have contains games of different NBA temas,
#Lets focus on one team to produce a time-series graph
#Let's focus on the Pistons first
#Create a df that extracts all the data from NBA_Games only when the NICKNAME  = Pistons, SEASON_ID = 22017 , and GAME_DATE is >= 2017-10-17
Pistons_Games = NBA_Games[(NBA_Games.NICKNAME == 'Pistons') & (NBA_Games.SEASON_ID == 22017) & (NBA_Games.GAME_DATE >= '2017-10-17')]

#Just for fun lets do the rest of the analysis for the Toronto Raptors
Raptors_Games = NBA_Games[(NBA_Games.NICKNAME == 'Raptors') & (NBA_Games.SEASON_ID == 22017) & (NBA_Games.GAME_DATE >= '2017-10-17')]

#Now plot the points earned by Raptors by time
Raptors_Games.plot(x='GAME_DATE' , y='PTS')
plt.savefig ('RAPTORS_PTS_TIME.png')

#Correlation Analysis
#We can detect the relationship between 2 variables in a scatterplot
#Lets use the number of assists and number of field goals made as an example
NBA_Games.plot.scatter(x='AST' , y='FGM')
Raptors_Games.plot.scatter(x='AST' , y='FGM')

#We can use the functions in seaborn to graph relationships between 2 variables
import seaborn as sns
sns.regplot(x='AST' , y='FGM' , data=Raptors_Games, marker='.') #regplot plots 2 variables as well as a regression line
plt.xlabel('Assits')
plt.ylabel('Field Goals Made')
plt.title('Relationship between the Numbers of Assists and Field Goals Made' , fontsize = 15)
#We can see a positive relationship/ correlation between the two variables

#Correlation Coefficient
#We can quantify the linear correlation by a correlation coefficient(CC). 
#The CC measures the joint variabiltiy of 2 random vars 
Raptors_Games['AST'].corr(Raptors_Games['FGM']) #CC of 0.642

#Now lets investigate the relationship between # of assits and # of field goals attempted
sns.regplot(x = 'AST' , y = 'FGA' , data=Raptors_Games , marker='.')
plt.xlabel('Assists')
plt.ylabel('Field Goals Attempted')
plt.title('Relationship between the Number of Assists and Field Goals Attempted' , fontsize = 15)

Raptors_Games['AST'].corr(Raptors_Games['FGA']) #CC of 0.22 

#We can use the lmplot() function to plot data and adjust the hue
#lmplot() combines regplot() and FacetGrid
#FacetGrid produces multi-plot grid for plotting conditional relationships.
#Thus FacetGrid allows us to separae a dataset into multiple panels based on specific conditions to visualize the relationship between multiple vars
sns.lmplot(x='AST' , y='FGM' , hue = 'WL' , data = Raptors_Games)
plt.xlabel('Assists')
plt.ylabel('Field Goals Made')
plt.title('Relationship between the Number of Assists and Field Goals Made' , fontsize = 15)


Raptors_Games['AST'].corr(Raptors_Games['FGA'], method = 'pearson') #We can also specify the method of the corr function to Pearson correlation coefficient

#%% Assignment Part 1 - Data Coding and Merging
import pandas as pd
import numpy as np
#Import the “NHL_Team.csv” data file and name the dataframe as “NHL_Team”
NHL_Teams = pd.read_csv('C:\\Users\William Zhao\\Desktop\\Data Science Learning\\Data\\NHL_team.csv')

#a) Delete the following variables: “Unnamed:0”, “abbr”, “tname”, “lname”, and “sname”.
NHL_Teams.drop(['Unnamed: 0' , 'abbr' , 'tname' , 'lname' , 'sname'] ,  inplace = True , axis = 1)

#b) Rename the variable “name” to “team_name”.
NHL_Teams = NHL_Teams.rename(columns ={'name' : 'team_name'})

#Import the “NHL_competition.csv” data file and name the dataframe as “NHL_Competition” in Jupyter Notebook.
NHL_Competition = pd.read_csv('C:\\Users\William Zhao\\Desktop\\Data Science Learning\\Data\\NHL_competition.csv')

#a) Delete the following variables: “Unnamed: 0”, “tz”, “start”, and “end”
NHL_Competition.drop(['Unnamed: 0' , 'tz' , 'start' , 'end'] , inplace = True , axis = 1)

#b) Rename the variable “name” to “competition_name”.
NHL_Competition = NHL_Competition.rename(columns = {'name' : 'competition_name'})

#Import the “NHL_game.csv” data file and name the dataframe as “NHL_Game” in Jupyter Notebook.
NHL_Game = pd.read_csv('C:\\Users\William Zhao\\Desktop\\Data Science Learning\\Data\\NHL_game.csv')

#a) Delete the following variables: “X”, “period”, and “status”.
NHL_Game.drop(['X' , 'period' , 'status'] , inplace = True , axis=1)

#Merge the dataframe “NHL_Team” into the dataframe “NHL_Game” by “tid.” Continue to name the merged dataframe as “NHL_Game.”
NHL_Game = pd.merge(NHL_Game , NHL_Teams, on = ['tid'])

#Merge the dataframe “NHL_Competition” into the dataframe “NHL_Game” by “comp_id.” Continue to name the merged dataframe as “NHL_Game.”
NHL_Game = pd.merge(NHL_Game , NHL_Competition , on = ['comp_id'])

#In the merged “NHL_Game” dataframe, create a variable “hgd” to indicate the goal difference between home and away score (hscore – ascore) and delete observations with missing value in the variable “hgd”.
NHL_Game['hgd'] = NHL_Game['hscore'] - NHL_Game['ascore'] 
NHL_Game = NHL_Game[pd.notnull(NHL_Game['hgd'])]

#Drop all observations with missing values, if there is still any, from the “NHL_Game” dataframe.
NHL_Game = NHL_Game[pd.notnull(NHL_Game)]

#Convert the type of the “date” variable from “object” to “datetime.”
import datetime
NHL_Game['date'] = pd.to_datetime(NHL_Game['date'])

#Sort the NHL games by “date” and show the first 15 observations.
NHL_Game.sort_values(by = ['date'] , ascending = [True]).head(15)

#Create two dataframes that separate the “NHL_Game” dataframe by home and away games. Name them “NHL_Home” and “NHL_Away”, respectively.
NHL_Home = NHL_Game[(NHL_Game.home_away == 'home')]
NHL_Away = NHL_Game[(NHL_Game.home_away == 'away')]

#a) Rename variables:
     #i) For away games, rename “ascore” to “goals_for”; rename “hscore” to “goals_against”
NHL_Away = NHL_Away.rename(columns = {'ascore' : 'goals_for' , 'hscore' : 'goals_against', 'team_name' : 'team'})

     #ii) For home games, rename “hscore” to “goals_for”; rename “ascore” to “goals_against” 
NHL_Home = NHL_Home.rename(columns = {'hscore' : 'goals_for' , 'ascore' : 'goals_against' , 'team_name' : 'team'})

#b) Create a “win” variable that equals to 1 if the team won the game; 0 if the team lost the game; and 0.5 if it was a draw. 
NHL_Home['win'] = np.where(NHL_Home['goals_for'] > NHL_Home['goals_against'] , 1 , 0)
NHL_Away['win'] = np.where(NHL_Away['goals_for'] > NHL_Away['goals_against'] , 1, 0)

#Append the “NHL_Home” and “NHL_Away” dataframes to be the new “NHL_Game” dataframe.

NHL_Game = pd.concat([NHL_Away, NHL_Home])

#Generate a team level dataframe that aggregates the total number of games won, the total number of “goals_for” and “goals_against” for each team in each competition (i.e. grouped by tid, competition_name and type). Name this new dataframe “NHL_Team_Stats”. Make sure to convert the indexes of the new dataframe back as  variables. 
NHL_Team_Stats= NHL_Game.groupby(['tid' , 'competition_name' , 'type'])[['win' , 'goals_for' , 'goals_against']].sum().reset_index()

#Create a dataframe “NHL_Game_Count” that include the total number of games played by each team in each competition (i.e. grouped by tid, competition_name and type). Name this new variable in the dataframe “game_count”.
NHL_Game_Count = NHL_Game.groupby(['tid' , 'competition_name' , 'type']).size().reset_index(name = 'GAME_COUNT')

#Merge dataframes.
#a) Merge the “NHL_Game_Count” dataframe into the “NHL_Team_Stats” dataframe by “tid”, “competition_name”, and “type”. Continue to name the merged dataframe “NHL_Team_Stats”.
NHL_Team_Stats = pd.merge(NHL_Game_Count , NHL_Team_Stats , on = ['tid' , 'competition_name' , 'type'])

#b) Merge the “NHL_Team” dataframe into the “NHL_Team_Stats” dataframe by “tid”. Continue to name the merged dataframe “NHL_Team_Stats”.
NHL_Team_Stats = pd.merge(NHL_Teams , NHL_Team_Stats , on = ['tid'])

#Import the “pp.pk.ppgf.csv” data file and name the dataframe as “NHL_PPPK” in Jupyter Notebook. Merge the “NHL_PPPK” dataframe into the “NHL_Team_Stats” dataframe by “tricode” and “competition_name”.
NHL_PPPK = pd.read_csv('C:\\Users\William Zhao\\Desktop\\Data Science Learning\\Data\\pp.pk.ppgf.csv')

NHL_Team_Stats = pd.merge(NHL_PPPK , NHL_Team_Stats , on = ['tricode' , 'competition_name'])

#Create new variables in the “NHL_Team_Stats” dataframe.

#a) Winning percentage (“win_pct”)=”win”/ total number of games played
NHL_Team_Stats['wpc'] = NHL_Team_Stats['win'] / NHL_Team_Stats['GAME_COUNT']

#b) Average goals for per game (“avg_gf”)=total number of goals for / total number of games played     
NHL_Team_Stats['avg_gf'] = NHL_Team_Stats['goals_for'] / NHL_Team_Stats['GAME_COUNT']

#c) Average goals against per game (“avg_ga”)=total number of goals against / total number of games played
NHL_Team_Stats['avg_ga'] = NHL_Team_Stats['goals_against'] / NHL_Team_Stats['GAME_COUNT']

#In the “NHL_Competition” dataframe, the variable “type” indicates the type of competition: type=2 – regular season. Create a dataframe that contains team statistics for games only during regular seasons. Name this dataframe “NHL_Team_R_Stats”. 
NHL_Team_R_Stats = NHL_Team_Stats[(NHL_Team_Stats.type == 2)]

#%% Assignment Part 2 : Calculating Summary Statistics

#In the “NHL_Game” dataframe, calculate summary statistics for the  “goals_for” variable; calculate summary statistics for the “goals_against” variable based on whether it is home or away game.
NHL_Game['goals_for'].describe()
NHL_Game.groupby(['home_away'])['goals_for'].describe()
NHL_Game.groupby(['home_away'])['goals_against'].describe()
    
#Create a histogram of the “goals_against” variable by whether the game is home or away 
#a) Make the color of the histogram green
#b) Set the number of bins to be 20
#c) Make sure the two sub-histograms share the same ranges for the x-axis and y-axis.
NHL_Game.hist(by = 'home_away' , column = 'goals_against' , color = 'green' , bins = 20 , sharex = True , sharey = True)
#%% Assignment Part 3: Correlation Analyses
import seaborn as sns
import matplotlib.pyplot as plt

#In the “NHL_Team_R_Stats” dataframe, make a scatter plot to depict the relationship between the total number of goals for and the winning percentage. 
#a) Plot the total number of goals for on the x-axis and winning percentage on the y-axis.
#b) Add a regression line to the scatter plot.
#c) Make the title of the graph “Relationship between Goals for and Winning Percentage” and make the font size 11.
#d) Label the x-axis “Total Goals for” and label the y-axis “Winning Percentage”.
sns.regplot(x = 'goals_for' , y = 'wpc' , data = NHL_Team_R_Stats , marker = '.')
plt.xlabel('Total Goals For')
plt.ylabel('Winning Percentage')
plt.title('Relationship between Goals for and Winning Percentage' , fontsize = 11)

#In the “NHL_Team_R_Stats” dataframe, calculate the correlation coefficient between total number of goals for and winning percentage.
NHL_Team_R_Stats['goals_for'].corr(NHL_Team_R_Stats['wpc']) #CC of 0.303

#Create a scatter plot of the total number of goals for and winning percentage similar to step 1 for regular season games. In this graph, group observations by “competition_name”.
#a) Plot the total number of goals for on the x-axis and winning percentage on the y-axis.
#b) Add a regression line to the scatter plot.
#c) Make the title of the graph “Relationship between Goals for and Winning Percentage” and make the font size 11.
#d) Label the x-axis “Total Goals for” and label the y-axis “Winning Percentage”.
sns.lmplot(x = 'goals_for' , y = 'wpc' , hue = 'competition_name' , data = NHL_Team_R_Stats)
plt.xlabel('Total Goals For')
plt.ylabel('Winning Percentage')
plt.title('Relationship between Goals for and Winning Percentage' , fontsize = 11)
#Notice that 2011 and 2012 are outliers in that the number of goals are much lower than the other regular seasons

#For the “NHL_Team_R_Stats” dataframe, delete observations of 2011 and 2012 seasons. Continue to name the dataframe “NHL_Team_R_Stats”.
NHL_Team_R_Stats = NHL_Team_R_Stats[NHL_Team_R_Stats.competition_name !='2011 NHL Regular Season']
NHL_Team_R_Stats = NHL_Team_R_Stats[NHL_Team_R_Stats.competition_name !='2012 NHL Regular Season']

#In the new “NHL_Team_R_Stats” dataframe, create a scatter plot of total number of goals for and winning percentage.
#a) Plot the total number of goals for on the x-axis and winning percentage on the y-axis.
#b) Add a regression line to the scatter plot.
#c) Make the title of the graph “Relationship between Goals for and Winning Percentage” and make the font size 11.
#d) Label the x-axis “Total Goals for” and label the y-axis “Winning Percentage”.
sns.regplot(x = 'goals_for' , y = 'wpc' , data = NHL_Team_R_Stats , marker = '.')
plt.xlabel('Total Goals For')
plt.ylabel('Winning Percentage')
plt.title('Relationship between Goals for and Winning Percentage' , fontsize = 11)
#Notice a much stronger correlation now that the 2011 and 2012 games are removed

#Calculate the correlation coefficient between total number of goals for and winning percentage in the updated “NHL_Team_R_Stats” dataframe.
NHL_Team_R_Stats['goals_for'].corr(NHL_Team_R_Stats['wpc']) # Updated CC of 0.7685

#Save dataframes as csv files. 
#a) Name the updated “NHL_Game” dataframe as “NHL_Game2”.
#b) Name the “NHL_Team_Stats” dataframe as “NHL_Team_Stats”.
#c) Name the “NHL_Team_R_Stats” dataframe as “NHL_Team_R_Stats”.\
#d) Make sure to exclude the index as a column in the csv files.

NHL_Game.to_csv('C:\\Users\William Zhao\\Desktop\\Data Science Learning\\Data\\NHL_Game2.csv' , index=False)
NHL_Team_Stats.to_csv('C:\\Users\William Zhao\\Desktop\\Data Science Learning\\Data\\NHL_Team_Stats.csv' , index=False)
NHL_Team_R_Stats.to_csv('C:\\Users\William Zhao\\Desktop\\Data Science Learning\\Data\\NHL_Team_R_Stats.csv' , index=False)






