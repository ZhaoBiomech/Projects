# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 21:16:13 2024

@author: William Zhao
"""
#%% Pythagorean Expectation & MLB

#Import Packages
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import matplotlib as plt
import seaborn as sns

#Import MLB data

MLB = pd.read_excel('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Retrosheet MLB game log 2018.xlsx')
print(MLB.columns.tolist()) #print a list of variable names

MLB # Typing in dataframe name shows us what it looks like

#Create a new dataframe that includes just the data we need
MLB18 = MLB [['VisitingTeam','HomeTeam','VisitorRunsScored','HomeRunsScore','Date']] #Create new dataframe with just the specified columns
MLB18 = MLB18.rename(columns = {'VisitorRunsScored' : 'VisR', 'HomeRunsScore' : 'HomR'}) # Rename the defined columns
MLB18

#We need to determine who won the game by creating new variables
#Wins indicated by 1 and losses indicated by 0
MLB18['hwin'] = np.where(MLB18['HomR']>MLB18['VisR'],1,0) # home win defined by when the home team runs are greater than visiting team runs
MLB18['awin'] = np.where(MLB18['HomR']<MLB18['VisR'],1,0) #away win defined by when the home team runs are less than visiting team runs
MLB18['count'] = 1 # We also create a 'counter' variable for each row to count the number of total games
MLB18

#Currently our df is a list of games, but we want a list of runs scored anc conceded by each team and its win percentage
#To do this, we will create 2 new dfs for home teams and away teams
MLBhome = MLB18.groupby('HomeTeam')[['hwin', 'HomR', 'VisR', 'count']].sum().reset_index() # Group by home team to obtain the sum of wins and runs and also the counter variable
MLBhome = MLBhome.rename(columns = {'HomeTeam' : 'team' , 'VisR' : 'VisRh' , 'HomR' : 'HomRh', 'count' : 'Gh'})
MLBhome

#Do the same for away games
MLBaway = MLB18.groupby('VisitingTeam')[['awin' , 'HomR' , 'VisR' , 'count']].sum().reset_index()
MLBaway = MLBaway.rename(columns={'VisitingTeam':'team','VisR':'VisRa','HomR':'HomRa','count':'Ga'})
MLBaway

#Now merge the 2 dfs together to have a list of all clubs with home and away records for the 2018 season
#We will overwrite our original df
MLB18 = pd.merge(MLBhome,MLBaway,on='team') #merge using the teams term 
MLB18

#Now create the total winnings, games played, runs scored and runs conceded by summing the home and away teams
MLB18['W'] = MLB18['hwin'] + MLB18['awin'] #Total wins 
MLB18['G'] = MLB18['Gh'] + MLB18['Ga']     #Total games played
MLB18['R'] = MLB18['HomRh'] + MLB18['VisRa'] #Total runs scored
MLB18['RA'] = MLB18['VisRh'] + MLB18['HomRa'] #Total runs against
MLB18

#Last step to prepare the data is to define win percentage and the pythagorean expectation
MLB18['wpc'] = MLB18['W'] / MLB18['G'] #Win percentage 
MLB18['pyth'] = MLB18['R']**2 / (MLB18['R']**2 + MLB18['RA']**2) #Pythagorean Expectation
MLB18

#Now ready to examine the data by plotting it as a scatter plot using the Seaborn package
sns.relplot(x = 'pyth' , y = 'W' , data = MLB18)

#Finally we generate a simple linear regression
pyth_lm = sm.ols(formula = 'wpc ~ pyth' , data=MLB18).fit()
pyth_lm.summary()

#%% Pythagorean Expectation & The IPL
#We are going to do the same sort of steps we did for the MLB, except now with the IPL
#This will highlight how differences in scoring between sports effect the Pythagorean Expectation

import pandas as pd
import numpy as np
import statsmodels.formula.api as sm 
import matplotlib.pyplot as plt
import seaborn as sns

#Import IPL Data
IPL18 = pd.read_excel("C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\IPL2018teams.xlsx")
IPL18

#First we identify when the home team is the winning team, and when the vising team is the winner
#Next we identify the runs scored by home team and the away team
#Finally we include a counter which we can add up to give total number of games for each team
IPL18['hwin'] = np.where(IPL18['home_team'] == IPL18['winning_team'],1,0) #Creates a new column that indicates when the text in home_team equals the text in winning_team shown by a 1
IPL18['awin'] = np.where(IPL18['away_team'] == IPL18['winning_team'],1,0) # Creates a new column that indicates when the away team wins
IPL18['htruns'] = np.where(IPL18['home_team'] == IPL18['inn1team'] , IPL18['innings1'] , IPL18['innings2']) # When the home team is the first inning team, return the first inning hits, or the 2nd inning hits if not 
IPL18['atruns'] = np.where(IPL18['away_team'] == IPL18['inn1team'] , IPL18['innings1'] , IPL18['innings2']) # Same as above but for away team
IPL18['count'] = 1
IPL18

#Now we want to group our data to aggregate the performance of home teams during the season.
IPLhome = IPL18.groupby('home_team')[['count','hwin','htruns','atruns']].sum().reset_index() # Create a new home team dataset grouped by home team and sums the data
IPLhome = IPLhome.rename(columns = {'home_team':'team','count':'Ph','htruns':'htrunsh','atruns':'atrunsh'}) # rename the columns
IPLhome

#Do the same for the away team data
IPLaway = IPL18.groupby('away_team')[['count','awin','htruns','atruns']].sum().reset_index() # Create an away team data set grouped by away team
IPLaway = IPLaway.rename(columns = {'away_team':'team','count':'Pa','htruns':'htrunsa','atruns':'atrunsa'}) #r rename the columns
IPLaway

#Merge the two dfs to obtain a full record for each team across the season
IPL18 = pd.merge(IPLhome,IPLaway,on=['team']) # Merge IPLhome and IPL away datasets by using "team"  
IPL18

#Now we need to aggreagate the home and away data for wins, games played and runs
IPL18['W'] = IPL18['hwin'] + IPL18['awin'] #Total wins
IPL18['G'] = IPL18['Ph'] + IPL18['Pa'] # Total games played
IPL18['R']= IPL18['htrunsh'] + IPL18['atrunsa'] #Total runs hit by adding home team runs hit at home plus away team runs hit away
IPL18['Ra'] = IPL18['atrunsh'] + IPL18['htrunsa'] #Total runs hit against by away team runs hit away plus home team runs hit away
IPL18

#Last step is to create the variables of win percentage and pythagorean expectation
IPL18['wpc'] = IPL18['W'] / IPL18['G']
IPL18['pyth'] = IPL18['R']**2 / (IPL18['R']**2 + IPL18['Ra']**2)
IPL18

#We can not plot the data
sns.relplot(x='pyth' , y='wpc', data=IPL18)

#Run a simple regression on the data
pyth_lm = sm.ols(formula = 'wpc ~ pyth' , data=IPL18).fit()
pyth_lm.summary()

#%% Pythagorean Expectation & NBA
#We will do the same now for NBA games, where it will be similar to the MLB in terms of data points

#Import Packages
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

#Import NBA data
NBA = pd.read_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\NBA_Games.csv')

#The data consists of games played between 2013 and 2019. Additionally, each game appears in two rows -> one for each row where each pair are mirror images of each other
#We want to extract games from just the 2018 season first
NBA18 = NBA[NBA.SEASON_ID == 22018] #Create a new df that just includes data from the NBA df when the SEASON_ID = 22018
NBA18.describe()

#The game result is in the column "WL". We need to create a variable that will indicate when a team won or lost
NBA18['result'] = np.where(NBA18['WL'] == 'W',1,0) # Create a new column for result where the value is 1 whenever WL = W

#For Pythagorean Expectation we only need the result, points scored and points conceded
NBA18teams = NBA18.groupby('TEAM_NAME')[['result' , 'PTS' , 'PTSAGN']].sum().reset_index()
NBA18teams

#Now we can calculate the Pythagorean expectation based on an 82 game season
NBA18teams['wpc'] = NBA18teams['result'] / 82
NBA18teams['pyth'] = NBA18teams['PTS']**2 / (NBA18teams['PTS']**2 + NBA18teams['PTSAGN']**2)
NBA18teams

#Run a simple linear regression on the data
pyth_lm = sm.ols(formula = 'wpc ~ pyth' , data = NBA18teams).fit()
pyth_lm.summary()

#%% Pythagorean Expectation and English Football
#Soccer hightlights a unique situation where draws are possible. This will require us to assign draws a value of 0.5
#We will also be looking at teams across different divisions 

import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

#Load the data
ENG18 = pd.read_excel('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Engsoccer2017-18.xlsx')
print(ENG18.columns.tolist())

#Our data is in the form of game results, so first we identify who won and lost or if it was a draw. We also create a counting variable
ENG18['hwinvalue']=np.where(ENG18['FTR']=='H',1,np.where(ENG18['FTR']=='D',0.5,0)) #Return a value of 1 when the home team one, or 0.5 if it was a draw, or 0 if it was a home-team loss
ENG18['awinvalue']=np.where(ENG18['FTR']=='A',1,np.where(ENG18['FTR']=='D',0.5,0))
ENG18['count']=1

#We must create 2 dfs to calculate home team and away team performance 
ENGhome = ENG18.groupby(['HomeTeam','Div'])[['count','hwinvalue','FTHG','FTAG']].sum().reset_index() #Create a new df grouped by home team and division
ENGhome = ENGhome.rename(columns={'HomeTeam':'team' , 'count':'Ph' , 'FTHG' : 'FTHGh' , 'FTAG' : 'FTAGh'}) #Rename the columns of the dataframe

ENGaway = ENG18.groupby(['AwayTeam'])[['count','awinvalue','FTHG','FTAG']].sum().reset_index() #Create a mirror df for the away team without grouping by Division for ease of merging after
ENGaway = ENGaway.rename(columns={'AwayTeam':'team' , 'count':'Pa' , 'FTHG' : 'FTHGa' , 'FTAG' : 'FTAGa'})

#Now merge the dfs 
ENG18 = pd.merge(ENGhome , ENGaway , on = ['team'] ) #Merge the 2 dfs by the column values in "team"

#Now sum the results by home and away measures to get the team overall performance for the season
ENG18['W'] = ENG18['hwinvalue'] + ENG18['awinvalue']
ENG18['G'] = ENG18['Ph'] + ENG18['Pa']
ENG18['GF'] = ENG18['FTHGh'] + ENG18['FTAGa']
ENG18['GA'] = ENG18['FTAGh'] + ENG18['FTHGa']
ENG18

#Now we calculate win percentage and pythagorean expectation
ENG18['wpc'] = ENG18['W'] / ENG18['G']
ENG18['pyth'] = ENG18['GF']**2 / (ENG18['GF']**2 + ENG18['GA']**2)

#Plot the data
sns.relplot(x='pyth' , y='wpc', data=ENG18, hue = 'Div') # Plot the data with a colour difference between divisions

#Run a simple linear regression on the data
pyth_lm = sm.ols(formula = 'wpc ~ pyth' , data = ENG18).fit()
pyth_lm.summary()

#%% Pythagorean Expectation As A Predictor in the MLB
#To use Pythagorean Expectation to predict team performance, we will follow similar steps to the first MLB analysis, but split the data in half at the All-Star Game
#Additionally, we can benchmark our prediction against the simplest forecast: win percentage will stay the same

import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

#Import the data
MLB = pd.read_excel('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Retrosheet MLB game log 2018.xlsx') 
print(MLB.columns.tolist())

#Create a df that only includes the data we need and a counter to count games
MLB18 = MLB[['VisitingTeam' , 'HomeTeam' , 'VisitorRunsScored' , 'HomeRunsScore' , 'Date']]
MLB18 = MLB18.rename(columns = {'VisitorRunsScored' : 'VisR' , 'HomeRunsScore' : 'HomR'})
MLB18['count'] = 1

#Create a df that records team performance as home team 
#We create an additional column 'home' which has a value 1 to designate that these were home team games
MLBhome = MLB18[['HomeTeam' , 'HomR' , 'VisR' , 'count', 'Date']].copy() # Create a new data frame that copies the data from a previous df
MLBhome['home'] = 1
MLBhome = MLBhome.rename(columns = {'HomeTeam' : 'team' , 'VisR' : 'RA' , 'HomR' : 'R'})

#Create a df that records team performance as away team
#Create a 'home' column which has a value 0 to designate that these were away games
MLBaway = MLB18[['VisitingTeam' , 'HomR' , 'VisR' , 'Date' , 'count']].copy()
MLBaway['home'] = 0
MLBaway = MLBaway.rename(columns = {'VisitingTeam' : 'team' , 'VisR' : 'R' , 'HomR' : 'RA'})

# *Here is where the method differs from previous versions. Instead of taking sums and averages, we first
# concatenate, meaning that we stack performances as home team and away team. This creates a list of games played
# by each team across the season.

MLB18 = pd.concat([MLBhome , MLBaway]) #Concatenate the two dfs together. The data is not merged and added together

#Now we define a win
MLB18['win'] = np.where(MLB18['R'] > MLB18['RA'] , 1 , 0) #Define a win where R > RA, indicated by a 1

#Now we define the season up to the All Star Game which was on July 17 2018 
Half1 = MLB18[MLB18.Date < 20180717] #Define Half1 as every date before 20180717
Half1.describe()

Half2 = MLB18[MLB18.Date > 20180717]
Half2.describe()

#Now we sum the number of games, wins, runs, and runs against for the 1st half of the season
Half1perf = Half1.groupby('team')[['count' , 'R' , 'RA' , 'win']].sum().reset_index()
Half1perf = Half1perf.rename(columns={'count' : 'count1' , 'win' : 'win1' , 'R' : 'R1' , 'RA' : 'RA1'})

#From this, we calculate the win percentage and Pythagorean expectation for the 1st half of the season
Half1perf['wpc1'] = Half1perf['win1'] / Half1perf['count1']
Half1perf['pyth1'] = Half1perf['R1']**2 / (Half1perf['R1']**2 + Half1perf['RA1']**2)

#Now we do the same for the 2nd half of the season
Half2perf = Half2.groupby('team')[['count' , 'R' , 'RA' , 'win']].sum().reset_index()
Half2perf = Half2perf.rename(columns = {'count' : 'count2' , 'win' : 'win2' , 'R' : 'R2' , 'RA' : 'RA2'})

Half2perf['wpc2'] = Half2perf['win2'] / Half2perf['count2']
Half2perf['pyth2'] = Half2perf['R2']**2 / (Half2perf['R2']**2 + Half2perf['RA2']**2) 

#Now we merge the 2 dfs
Half2predictor = pd.merge(Half1perf , Half2perf , on = 'team')

#First we plot Pyth Expectation vs win percentage in the second half of the season
sns.relplot(x='pyth1' , y='wpc2' , data= Half2predictor)

#Now compare to a plot of win percentage1 vs win percentage2
sns.relplot(x='wpc1' , y='wpc2' , data=Half2predictor)

#The 2 plots are quite similar, but we can be more precise by comparing the correlation coefficients
#The first row shows the correlation of win percentage in 2nd half of the season against the other variables
#We want to focus on the 3rd and 4th columns
keyvars = Half2predictor[['wpc2' , 'wpc1' , 'pyth1' , 'pyth2']]
keyvars.corr()

keyvars = keyvars.sort_values(by=['wpc2'] , ascending = False)
keyvars

#We can see from the correlation matrix that win percentage in the second half 
#of the season is correlated with win percentage in the first half of the season - 
#the correlation coefficient is +0.653. It's not surprising that performance in the 
#first half of the season is to an extent predictive of performance in the second half. 
#But there are also clearly things that can change.
#When we sort the teams from highest to lowest send half of season win percentage, 
#we find a mixed picture. Some clubs perform with less than one percentage point difference 
#in each half, e.g. The Brave (ATL), the Padres (SDN) or the Orioles (BAL), while others differed 
#by more than ten percentage points, e.g. the Rays (TBA), the Mets (NYN) or the Mariners (SEA).
#We could simply use first half win percentage as a predictor of second half win percentage, 
#but when we look at the correlation matrix we can see that the Pythagorean Expectation is an even 
#better forecast - the correlation coefficient is higher, at +0.691. To be sure, the difference is 
#not large, but it is slightly better. This was, in fact, the initial impetus for Bill James when
#introducing the statistic. He argued that a win could ride on lucky hit and the difference of just 
#one run, which made wins a less reliable predictor than the aggregate capacity to produce runs and 
#limit conceding runs. As in many aspects of baseball analysis, our data show that James was quite right.

#%% Week 1 Assignment: Pythagorean Expecatation as a Predictor for EPL 
#Code is written below the steps given

import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

#Load the datafile (this contains 6 variables: the date, home team, away team, 
#goals scored (FTHG), goals against (FTAG) andthe result (H- home win, D- draw, A – away win).
EPL = pd.read_excel('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\EPL2017-18.xlsx')

#Create a value for a home wins (win= 1, draw=0.5, loss= 0) and away wins and a count variable for each game (=1).
EPL['homewin'] = np.where(EPL['FTR'] == 'H' , 1, np.where(EPL['FTR'] == 'D' ,0.5,0))
EPL['awaywin'] = np.where(EPL['FTR'] == 'A' , 1, np.where(EPL['FTR'] == 'D' ,0.5,0))
EPL['count'] = 1
#Create a file for games played in 2017 (before date 20180000) and another one for games played in 2018 (after date 20180000).
EPL2017 = EPL[EPL.Date < 20180000]
EPL2018 = EPL[EPL.Date > 20180000]
EPL2018.describe()
#For the 2017 games, use .groupby to create a dataframe aggregating by home team the variables for count, home wins, goals for and goals against.
EPL2017home = EPL2017.groupby('HomeTeam')[['count' , 'homewin' , 'FTHG' , 'FTAG']].sum().reset_index()
 
#Then, use .groupby to  create a separate dataframe aggregating by away team the variables for count, away wins, goals for and goals against.
EPL2017away = EPL2017.groupby('AwayTeam')[['count' , 'awaywin' , 'FTHG' , 'FTAG']].sum().reset_index()

#Rename the variables to denote whether they are aggregates for home team or away team.
EPL2017home = EPL2017home.rename(columns = {'HomeTeam' : 'team' , 'count' : 'Ph' , 'FTHG' : 'FTHGh' , 'FTAG' : 'FTAGh'})

EPL2017away = EPL2017away.rename(columns = {'AwayTeam' : 'team' , 'count' : 'Pa' , 'FTHG' : 'FTHGa' , 'FTAG' : 'FTAGa'})
#Then merge the home and away dataframes.
EPL2017 = pd.merge(EPL2017home , EPL2017away , on=['team'])

#Sum the values of home and away wins, games, goals for and goals against, then create the values for win percentage (wpc) and the Pythagorean expectation (pyth). 
EPL2017['W'] = EPL2017['homewin'] + EPL2017['awaywin']
EPL2017['G'] = EPL2017['Ph'] + EPL2017['Pa']
EPL2017['GF'] = EPL2017['FTHGh'] + EPL2017['FTAGa']
EPL2017['GA'] = EPL2017['FTAGh'] + EPL2017['FTHGa']

EPL2017['wpc1'] = EPL2017['W'] / EPL2017['G']
EPL2017['pyth1'] = EPL2017['GF']**2 / (EPL2017['GF']**2 + EPL2017['GA']**2)

#Now repeat steps 4-8 for the 2018 games. Be sure to give different names for wpc and pyth in 2017 and 2018.
EPL2018home = EPL2018.groupby('HomeTeam')[['count' , 'homewin' , 'FTHG' , 'FTAG']].sum().reset_index()
EPL2018away = EPL2018.groupby('AwayTeam')[['count' , 'awaywin' , 'FTHG' , 'FTAG']].sum().reset_index()

EPL2018home = EPL2018home.rename(columns = {'HomeTeam' : 'team' , 'count' : 'Ph' , 'FTHG' : 'FTHGh' , 'FTAG' : 'FTAGh'})
EPL2018away = EPL2018away.rename(columns = {'AwayTeam' : 'team' , 'count' : 'Pa' , 'FTHG' : 'FTHGa' , 'FTAG' : 'FTAGa'})

EPL2018 = pd.merge(EPL2018home , EPL2018away , on = ['team'])

EPL2018['W'] = EPL2018['homewin'] + EPL2018['awaywin'] 
EPL2018['G'] = EPL2018['Ph'] + EPL2018['Pa']
EPL2018['GF'] = EPL2018['FTHGh'] + EPL2018['FTAGa']
EPL2018['GA'] = EPL2018['FTAGh'] + EPL2018['FTHGa']

EPL2018['wpc2'] = EPL2018['W'] / EPL2018['G']
EPL2018['pyth2'] = EPL2018['GF']**2 / (EPL2018['GF']**2 + EPL2018['GA']**2)

#Now merge 2017 and 2018 summary files.
EPLpredictor = pd.merge(EPL2017 , EPL2018 , on = ['team'])

#Now generate a correlation matrix for the wpc and pyth variables for 2017 and 2018
sns.relplot (x='pyth1' , y='wpc2' , data=EPLpredictor)
sns.relplot (x='wpc1', y='wpc2' , data=EPLpredictor)

keyvars = EPLpredictor[['wpc2' , 'wpc1' , 'pyth1' , 'pyth2']]
keyvars.corr()

keyvars = keyvars.sort_values(by=['pyth2'] , ascending = False)
keyvars

#Purely for answering the assignment
EPL2018['gap']= EPL2018['homewin'] - EPL2018['awaywin']
EPL2017['diff'] = EPL2017['pyth1'] - EPL2017['wpc1']





























