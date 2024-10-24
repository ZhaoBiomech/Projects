# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 22:07:28 2024

@author: William Zhao
"""
#%% Basketball Heatmap
#In this part we are going to look at ways to visualize performance in basketball. Our analysis is going to 
#focus on the where the ball was thrown from, which is recorded using x,y coordinates. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import shot log data for the NBA season 2016/2017
shot = pd.read_csv('C:\\Users\\William Zhao\\Desktop\Data Science Learning\\Data\Files\\NBA Shotlog_16_17.csv')

#Use the pd.set_option to ensure that all columns are displayed when printing a dataframe, by setting it to 100
pd.set_option('display.max_columns' , 100)
print(shot.columns.tolist())
shot.describe()

#We can generate a simple plot to examine the location of shots being taken
x = shot['location_x']
y = shot['location_y']
plt.scatter(x,y, s=.005 , c='r' , marker = '.') #plot a scatter plot using x and y vars, marker size of 0.005,  red as the colour, and markers as dots, 
#This plot gives us a clear picture of the location of shots, however it does not take into account the size of the court

#We can generate a new plot that is scaled for court dimensions and with a grid added
plt.figure(figsize =(94/6 , 50/6)) # NBA courts are 94 x 50 feet, but we scale it by 6 to control the size of the plot
plt.scatter(x , y , s = 0.1 , c = 'r' , marker = '.') 
plt.minorticks_on() #Turn minorticks on
plt.grid(which = 'major' , linestyle = '-' , linewidth = '.5' , color = 'black') #We can add in major grid marks to identify more clearly the different locations
plt.grid(which = 'minor' , linestyle = ':' , linewidth = '0.5' , color = 'red') # Add in minor grid marks

#Rather thank looking at both ends of the court, we can just look at half by fixing the range
#of the x-axis using plt.xlim -> x-axis runs from 0 to 933 so we make the bounds from half court to the end
plt.figure(figsize = (94/12 , 50/6))
plt.scatter(x , y , s=0.1 , c = 'r' , marker = '.')
plt.minorticks_on()
plt.grid(which = 'major' , linewidth = '.5' , color = 'black')
plt.grid(which = 'minor' , linewidth = '0.5' , color = 'red')
plt.xlim(933/2, 933) 

#What we can also do is show ALL of the shots on just the right hand side of the court
#To do this we recode the location_x variables, where x< 933/2 (left hand side) as equal to 933 - location_x
#This produces a MIRROR IMAGE of the x coordinate in the right hand half of the court
#We also need to take the mirro image of the y-coordinate for shots which is 500-location_y
shot['halfcourt_x'] = np.where(shot['location_x'] < 933/2 , 933 - shot['location_x'] , shot['location_x']) #create a new variable where if location_x is less than 933/2, then return 933 - location_x, otherwise return the same value
shot['halfcourt_y'] = np.where(shot['location_y'] < 933/2 , 500 - shot['location_y'] , shot['location_y']) #create a new variable where if location_y is less than 933/2, then return 500 - location_y, otherwise return the same value
shot.describe()

#Now we can plot just the halfcourt
hx = shot['halfcourt_x']
hy = shot['halfcourt_y']
plt.figure(figsize = (94/12 , 50/6))
plt.scatter(hx , hy , s= 0.1 , c = 'r' , marker = '.')
plt.minorticks_on()
plt.grid(which = 'major' , linestyle = '-' , linewidth = '0.5' , color = 'black')
plt.grid(which = 'minor' , linestyle = ':' , linewidth = '0.5' , colour = 'red')
plt.title('Shots' , fontsize = 15)

#Now that all the shots are displayed on one half of the court, we can breakdown the shots into 3 categories : scored, missed and blocked

#Scoring Shots
Scored = shot[shot.current_shot_outcome == 'SCORED']
hxs = Scored['halfcourt_x']
hys = Scored['halfcourt_y']

plt.figure(figsize = (94/12 , 50/6))
plt.scatter(hxs , hys , s=0.1 , c='g' , marker = '.') #colour green
plt.minorticks_on()
plt.grid(which = 'major' , linestyle = '-' , linewidth = '0.5' , color = 'black')
plt.grid(which = 'minor' , linestyle = ':' , linewidth = '0.5' , color = 'red' )
plt.title('Scored' , fontsize = 15)

#Missed Shots
Missed = shot[shot.current_shot_outcome == 'MISSED']
hxm = Missed['halfcourt_x']
hym = Missed['halfcourt_y']

plt.figure(figsize = (94/12 , 50/6))
plt.scatter(hxm , hym , s=0.1 , c='b' , marker = '.') #colour blue
plt.minorticks_on()
plt.grid(which = 'major' , linestyle = '-' , linewidth = '0.5' , color = 'black')
plt.grid(which = 'minor' , linestyle = ':' , linewidth = '0.5' , color = 'red' )
plt.title('Scored' , fontsize = 15)

#Blocked Shots
Blocked = shot[shot.current_shot_outcome == 'BLOCKED']
hxb = Blocked['halfcourt_x']
hxy = Blocked['halfcourt_y']

plt.figure(figsize = (94/12 , 50/6))
plt.scatter(hxm , hym , s=0.1 , c='m' , marker = '.') #colour magenta
plt.minorticks_on()
plt.grid(which = 'major' , linestyle = '-' , linewidth = '0.5' , color = 'black')
plt.grid(which = 'minor' , linestyle = ':' , linewidth = '0.5' , color = 'red' )
plt.title('Scored' , fontsize = 15)

#While informative, the location of shots by shot type is not that surprising. What would be of greater interest is comparing of individual players
#We can do this easily by taking subsets as we did above.
playersn = shot.groupby('shoot_player')['current_shot_outcome'].describe().reset_index() #New df grouped by player 
playersn.sort_values(by = 'count' , ascending = False)

#We can compare 2 popular players like LeBron James and Steph Curry
LeBron= shot[shot['shoot_player'] == 'LeBron James']
Curry = shot[shot['shoot_player'] == 'Stephen Curry']

hxL = LeBron['halfcourt_x']
hyL= LeBron['halfcourt_y']
colors = np.where(LeBron['current_shot_outcome'] == 'SCORED' , 'r' , np.where(LeBron['current_shot_outcome'] == 'MISSED' , 'b' , 'g')) #Define colors where when SCORED its returns 'r', when MISSED it returns 'b', and when it's not(Blocked) it returns 'g'
plt.figure(figsize = (94/12 , 50/6))
plt.scatter(hxL , hyL , s = 10, c = colors , marker = '.')
plt.grid(True) #Turn on grid
plt.title('LeBron James' , fontsize = 15)

hxC = Curry['halfcourt_x']
hyC= Curry['halfcourt_y']
colors = np.where(Curry['current_shot_outcome'] == 'SCORED' , 'r' , np.where(Curry['current_shot_outcome'] == 'MISSED' , 'b' , 'g')) #Define colors where when SCORED its returns 'r', when MISSED it returns 'b', and when it's not(Blocked) it returns 'g'
plt.figure(figsize = (94/12 , 50/6))
plt.scatter(hxC , hyC , s = 10, c = colors , marker = '.')
plt.grid(True) #Turn on grid
plt.title('Steph Curry' , fontsize = 15)

#Now we can even put the two player plots side-by-side
f = plt.figure(figsize = (94/6 , 50/6)) #Matplotlib 'Figure' object to add a subplot
ax = f.add_subplot(121) # This adds a subplot to the figure 'f'. The argument '1222' is a shorthand notation that specifies the layout of the subplot -> '1' row in the subplot grid , '2' columns in the subplot grid , '1' index of the subplot you are adding
colors = np.where(LeBron['current_shot_outcome'] == 'SCORED' , 'r' , np.where(LeBron['current_shot_outcome'] == 'MISSED' , 'b' , 'g'))
ax = plt.scatter(hxL , hyL, s=10 , c = colors , marker = '.')
plt.grid(True)
plt.title('LeBron James' , fontsize = 15)

ax = f.add_subplot(122) # Add in the second subplot to the '2nd' index
colors = np.where(Curry['current_shot_outcome'] == 'SCORED' , 'r' , np.where(Curry['current_shot_outcome'] == 'MISSED' , 'b' , 'g'))
ax = plt.scatter(hxC , hyC , s=10 , c = colors , marker = '.')
plt.grid(True)
plt.title('Steph Curry' , fontsize = 15)

#As a final exercise, we can ZOOM IN on the data to see what it looks like up close. We can identify a particular location based on our grid
#First we look at shots from the paint , defined as x coordinates between 700 and 900, and y coordinates between 200 and 300
rect1 = shot[(((shot['location_x'] > 700) & (shot['location_x'] < 900)) & \
              ((shot['location_y'] > 200) & (shot['location_y'] < 300)))]

xr = rect1['halfcourt_x']
yr= rect1['halfcourt_y']
colors = np.where(rect1['current_shot_outcome'] == 'SCORED' , 'r' , np.where(rect1['current_shot_outcome'] == 'MISSED' , 'b' , 'g')) #Define colors where when SCORED its returns 'r', when MISSED it returns 'b', and when it's not(Blocked) it returns 'g'
plt.figure(figsize = (94/12 , 50/6))
plt.scatter(xr , yr , s = 10, c = colors , marker = '.')
plt.grid(True) 

#Note how the locations appear in vertical lines. This reflects the fact that the resolution of the location coding is finite
#The difference betwen adjacent vertical lines on this plot is approx 1 inch
#But we cn still zoom in further. We will look at the area immediately under the basket
rect2 = shot[(((shot['location_x'] > 850) & (shot['location_x'] < 875)) & \
              ((shot['location_y'] > 240) & (shot['location_y'] < 260)))]

xq = rect2['halfcourt_x']
yq = rect2['halfcourt_y']
colors = np.where(rect2['current_shot_outcome'] == 'SCORED' , 'r' , np.where(rect2['current_shot_outcome'] == 'MISSED' , 'b' , 'g')) #Define colors where when SCORED its returns 'r', when MISSED it returns 'b', and when it's not(Blocked) it returns 'g'
plt.figure(figsize = (94/12 , 50/6))
plt.scatter(xq , yq , s = 10, c = colors , marker = '.')
plt.grid(True)

#It turns out this degree of resolution is not very informative. The problem is that all the points are piled upon each other, and there is nothing in between

#%% Indian Premier League Graphs
#The goal of this section is to show how powerful graphs can be as tools for understanding sports data
#As a general rule, any data analysis should start with drawing some graphs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IPL2018 = pd.read_excel('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\IPL2018_results.xlsx')
pd.set_option('display.max_columns' , 50) #Change pd setting to display max column of 50

print(IPL2018.columns.tolist()) #print the name of the columns to a list to examine the variables

#The variables we are interested in are runs scored by each team. 
#A histogram will show us the variration of runs scored.
IPL2018.hist(column = 'innings2' , bins = 10) #We can adjust the number of bins to ssee what better represents the data
IPL2018.hist(column = 'innings1' , bins = 10)

#Comparing the 2 graphs, we can see that the x-axes are not the same ranges. We need to adjust
#to make a fair comparison
IPL2018.hist(column = 'innings2' , bins = 10) 
plt.xlim(60,250)
plt.ylim(0,20)

IPL2018.hist(column = 'innings1' , bins = 10)
plt.xlim(60,250)
plt.ylim(0,20)
plt.plot

#We can show the 2 distributions on the same histogram
#We can also specify the alpha, which is the degree of transparency 
IPL2018[['innings1' , 'innings2']].plot.hist(alpha = 0.5 , bins = 10) #Can play with the alpha to see what works best
plt.xlabel("Runs")
plt.ylabel('Frequency')
plt.xlim(60,250)
plt.ylim(0,20)

#Having looked at distributions, now let's compare histograms for the teams that win / lose
#First define winning and losing teams, derived by the number of runs scored

IPL2018['winscore'] = np.where(IPL2018['innings1'] > IPL2018['innings2'] , IPL2018['innings1'] , IPL2018['innings2']) #If innings1 is greater than innings2,  return innings1 score, if not return innings2 score
IPL2018['losescore'] = np.where(IPL2018['innings1'] > IPL2018['innings2'] , IPL2018['innings2'] , IPL2018['innings1'])

#Now we can plot these scores
IPL2018[['winscore' , 'losescore']].plot.hist(alpha = 0.5 , bins = 10)
plt.xlabel('Runs')
plt.ylabel('Frequency')
plt.title('Runs distribution by innings' , fontsize = 15)
plt.xlim((60,250))
plt.ylim((0,20))

#We can now look at a particular game, the opening game of 2018 between Mumbai Indians and Chennai Super Kings
MI_CSK = pd.read_excel('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\MIvCSKadj.xlsx')
print(MI_CSK.columns.tolist())

#We can plot a linechart showing the runs against delivery number for Mumbai, the team that batted first
plt.plot(MI_CSK['MI_delivery_no'] , MI_CSK['MI_runs_total_end'])

#We can then incorporate the fall of wickets into the chart, to see how their batting resources changed as the inning progressed
MIwicket = MI_CSK[MI_CSK['MI_wicket'] > 0] # Create a subset df from only when MI_wicket > 0
CSKwicket = MI_CSK[MI_CSK['CSK_wicket'] > 0]

#We can now plot the fall of wickets alongside the runs total
#Note that we obtain the red dots for wickets by specificying 'ro' = 'r' for red and 'o' for circle dots
plt.plot(MI_CSK['MI_delivery_no'] , MI_CSK['MI_runs_total_end'] , MIwicket['MI_runs_total_end'] , 'ro')

#We can now combine the runs scored profile for Mumbai with Chennai's
plt.plot(MI_CSK['CSK_delivery_no'] , MI_CSK['MI_runs_total_end'] , MI_CSK['CSK_runs_total_end']) #Add another line of data to the plot

#We can now plot the fall of wickets on the chart
plt.plot(MI_CSK['CSK_delivery_no'] , MI_CSK['MI_runs_total_end'] , MIwicket['MI_runs_total_end'] , 'bo')
plt.plot(MI_CSK['CSK_delivery_no'] , MI_CSK['CSK_runs_total_end'] , CSKwicket['CSK_runs_total_end'] , 'ro')

#We did this for one game, but we can write a program which will allow us to reproduce these profiles for
#any game in IPL2018 season, and compare profiles for any pair of games
#First load a dataframe that includes every delivery for the season
IPLbyb = pd.read_excel('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\IPLbyb.xlsx')
print(IPLbyb.columns.tolist())

#We are now going to define 2 functions: one that will allow to create a comparable chart for any game,
#and the other that will allow us to specify 2 games to compare

#First function for plotting the runs and wickets for each team in a game
def plot_runs_wickets(IPLbyb, ax):
    gameno = IPLbyb['gameno'].unique()[0]
    for inning, data in IPLbyb.groupby('innings_number'):
        #create seprate dataframe for wickets
        wicket = data[data['wicket'] > 0]
        #plots line
        ax.plot(data['delivery_no'] , data['runs_total_end'])
        #plots markers
        marker = 'bo' if inning == 1 else 'ro'
        ax.plot(wicket['delivery_no'], wicket['runs_total_end'] , marker)
        #labels
        ax.set_xlabel('balls')
        ax.set_ylabel('runs')
        ax.set_title(f'Game{gameno}') #F-string to include game number variable
    ax.legend(['runs1' , 'wkt1' , 'runs2' , 'wkt2'])

#Second, a function that allows us to plot two or more games at the same time
def plot_runs_wickets_multigame(list_games):
    n = len(list_games)
    fig, axs = plt.subplots(n , 1, figsize = (6,15))
    for i , gameno in enumerate(list_games):
        game = IPLbyb[IPLbyb['gameno'] == gameno]
        plot_runs_wickets(game, axs[i] if n > 1 else axs)

#These two functions will allow us to produce multiple charts to display games alongside each other
#But first we should generate a list of games so that we can decide which game's number refer to which teams
#Identify if the home team batted first
IPLbyb['hometeambatsfirst'] = np.where((IPLbyb['home team'] == IPLbyb['batting_team']) & (IPLbyb['innings_number'] == 1) , 'yes' , 'no')

#Drop duplicates so we just have a list of games
games = IPLbyb.drop_duplicates('gameno')

#Generate list of games
games = games[['gameno' , 'home team' , 'batting_team' , 'bowling_team' , 'hometeambatsfirst']]

#Create a new column called road team by using the apply function to apply a 
#Lambda function across each row (axis = 1)
#Lambda function takes row 'x' (batting_team) as input, then checks if the value in 'home_team' is equal to the value.  
games['road team'] = games.apply(lambda x: x['batting_team'] if x['home team'] == x['bowling_team'] else x['bowling_team'], axis = 1)

#This line of code is an assert statement to verify a condition about games df
#We want to compare and make sure that no home team = away team 
assert (games['home team'] !=games['road team']).all()

#Reduce df now to just know which team bats first and gameno
games = games[['gameno' , 'home team' , 'road team' , 'hometeambatsfirst']]

#We can see that game 27 is the rematch between the MI and CSK
#We can use our functions to enter the game numbers we want to compare
plot_runs_wickets_multigame([1,27])

#%% Baseball Heatmaps
#In this part we are going to be visualizing performance in baseball
#Specifically, we are going to use 'heatmaps' to show location and frequency of data
import pandas as pd
import matplotlib.pyplot as plt

#Import MLB playbyplay data from 2018
MLBAM18 = pd.read_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\MLBAM18.csv')
MLBAM18.drop(['Unnamed: 0'], axis = 1, inplace= True)
pd.set_option('display.max_columns' , 100)

print(MLBAM18.columns.tolist()) #The df contians lots of variables that we won't need for this exercise

#We will now restrict the data to a manageable set of variables
MLBmap = MLBAM18[['gameId' , 'home_team' , 'away_team' , 'stadium' , 'inning' , 'batterId' , 'batterName' ,\
                  'pitcherId' , 'pitcherName' , 'event' , 'timestamp' , 'stand' , 'throws' , 'x' , 'y' , 'our.x' , 'our.y']]

#We focus a good deal on different events, so it's useful to list them before we go any further
MLBmap['event'].unique() #We can use the .unique() command to see all the different possibilites

#First we can start with a simple scatter plot to examine the x and y coordinate data of balls hit
plt.scatter(MLBmap['our.x'] , MLBmap['our.y'] , s=.001 , c = 'r' , marker = '.')

#To further break it down, we can plot a heatmap of singles, doubles, triples, and HRs
#To do this, we first create new dfs for each specific event
Single = MLBmap[MLBmap.event == 'Single'] 
Double = MLBmap[MLBmap.event == 'Double']
Triple = MLBmap[MLBmap.event == 'Triple']
Homer = MLBmap[MLBmap.event == 'Home Run']

#Now we can plot an individual heatmap for each event
plt.scatter(Single['our.x'] , Single['our.y'] , s=.002 , c = 'darkorange' , marker = '.' )
plt.scatter(Double['our.x'] , Double['our.y'] , s = 1 , c= 'dodgerblue' , marker = '.')
plt.scatter(Triple['our.x'] , Triple['our.y'] , s = 1, c = 'g' , marker = '.')
plt.scatter(Homer['our.x'] , Homer['our.y'] , s = .20 , c = 'm' , marker = '.')

#Instead of plotting the 4 events separately, we can also plot them all together to make easier comparisons
#First we create them as 4 separate subplots
f = plt.figure(figsize = (15,3))
ax = f.add_subplot(141)
ax = plt.scatter(Single['our.x'] , Single['our.y'] , s = 0.1 , c = 'darkorange' , marker = '.')
plt.ylim((0,500))
ax = f.add_subplot(142)
ax = plt.scatter(Double['our.x'] , Double['our.y'] , s = 1 , c= 'dodgerblue' , marker = '.')
plt.ylim((0,500))
ax = f.add_subplot(143)
ax = plt.scatter(Triple['our.x'] , Triple['our.y'] , s = 1, c = 'g' , marker = '.')
plt.ylim((0,500))
ax = f.add_subplot(144)
ax = plt.scatter(Homer['our.x'] , Homer['our.y'] , s = .20 , c = 'm' , marker = '.')
plt.ylim((0,500))

#Now we can combine them onto one single plot
ax = plt.scatter(Single['our.x'] , Single['our.y'] , s=.002 , c = 'darkorange' , marker = '.' )
ax = plt.scatter(Double['our.x'] , Double['our.y'] , s = 1 , c= 'dodgerblue' , marker = '.')
ax = plt.scatter(Triple['our.x'] , Triple['our.y'] , s = 1, c = 'g' , marker = '.')
ax = plt.scatter(Homer['our.x'] , Homer['our.y'] , s = .20 , c = 'm' , marker = '.')

#Another way to use scatter diagrams is to compare at-bats which result in a hit,
#and at-bats that result in an out. First we generate the 2 scatter plots seperately, then create them alongside each other

#Outs
Outs = MLBmap[(MLBmap.event == 'Groundout') | (MLBmap.event == 'Flyout') | (MLBmap.event == 'Pop Out') | 
              (MLBmap.event == 'Forceout') | (MLBmap.event == 'Lineout') | (MLBmap.event == 'Grounded Into DP')]

plt.scatter(Outs['our.x'] , Outs['our.y'] , s=.01 , c='r' , marker = '.')

#Hits
Hits = MLBmap[(MLBmap.event == 'Single') | (MLBmap.event == 'Double') | (MLBmap.event == 'Triple')|
              (MLBmap.event == 'Home Run')]

plt.scatter(Hits['our.x'] , Hits['our.y'] , s=0.1 , c='b' , marker = '.')

#Hits vs. Outs Plot
f = plt.figure(figsize = (15,3))
ax = f.add_subplot(131)
ax = plt.scatter(Outs['our.x'] , Outs['our.y'], s=.01, c='r' , marker = '.')
ax2 = f.add_subplot(132)
ax2 = plt.scatter(Hits['our.x'] , Hits['our.y'] , s=.01 , c='b' , marker = '.')

ax3 = f.addsubplot(133)
ax3=plt.scatter(Outs['our.x'],Outs['our.y'], s=.01,c='r', marker= '.')
ax3=plt.scatter(Hits['our.x'],Hits['our.y'], s=.01,c='b', marker= '.')

#In this final part, we can compare stadiums. Ballparks do not have identical dimensions
#so we can look to see if the pattern of coordinates is different. 
stadiums = MLBmap.groupby('stadium')['gameId'].count().reset_index()

#Lets compare 3 ballparks, Tropicana Field (smallest) , Dodgers Stadium (largest) , and Rogers Centre (Go bluejays!)

#Tropicana Field
Trop = MLBmap[MLBmap.stadium == 'Tropicana Field']
plt.scatter(Trop['our.x'] , Trop['our.y'] , s = 1, c = 'r' , marker = '.')

#Dodger Stadium
Dodge = MLBmap[MLBmap.stadium == 'Dodger Stadium']
plt.scatter(Dodge['our.x'] , Dodge['our.y'] , s = 1 , c = 'dodgerblue' , marker = '.')


#Rogers Centre
Rogers = MLBmap[MLBmap.stadium == 'Rogers Centre']
plt.scatter(Rogers['our.x'] , Rogers['our.y'] ,s = 1 , c ='g' , marker = '.'  )

#All fields together and overlayed on top of each other
f = plt.figure(figsize =(15,3))
ax = f.addsubplot(141)
ax = plt.scatter(Trop['our.x'] , Trop['our.y'] , s = .5 , c = 'r' , marker = '.')
ax2 = f.add_subplot(142)
ax2 = plt.scatter(Dodge['our.x'] , Dodge['our.y'] , s = .5 , c = 'dodgerblue' , marker = '.')
ax3 = f.addsubplot(143)
ax3 = plt.scatter(Rogers['our.x'],Rogers['our.y'], s=.5,c='r', marker= '.')
ax4 = f.add_subplot(144)

ax4 = plt.scatter(Trop['our.x'] , Trop['our.y'] , s = .5 , c = 'r' , marker = '.')
ax4 = plt.scatter(Dodge['our.x'] , Dodge['our.y'] , s = .5 , c = 'dodgerblue' , marker = '.')
ax4 = plt.scatter(Rogers['our.x'],Rogers['our.y'], s=.5,c='r', marker= '.')

#We can also compare where players hit. Batters are identified as lefties or righties
#First we list all players, then choose a righty and lefty.

#Comparing players, first we use a pivot table to list players by at bat
playersn = MLBmap.groupby('batterId')['batterName'].describe().reset_index()
playersn.sort_values(by = 'count' , ascending = False) #Notice that Justin Turner and Nick Markakis are a top righty and lefty respectively

#Compare a Righty and Lefty
#Turner(R)
b607208 = MLBmap[MLBmap.batterId == 607208]
plt.scatter(b607208['our.x'] , b607208['our.y'] , s=10 , c='r' , marker = '.')

#Markakis(L)
b455976 = MLBmap[MLBmap.batterId == 455976]
plt.scatter(b455976['our.x'],b455976['our.y'], s=10,c='b', marker= '.')

#Turner and Markakis together
plt.scatter(b607208['our.x'],b607208['our.y'], s=10,c='r', marker= '.')
plt.scatter(b455976['our.x'],b455976['our.y'], s=10,c='b', marker= '.')

#We can also use a heatmap to compare ALL righties and lefties
#Lefties
Left = MLBmap[MLBmap.stand == 'L']
plt.scatter(Left['our.x'] , Left['our.y'] , s=.01 , c='r' , marker='.')

#Righties
Right = MLBmap[MLBmap.stand == 'R']
plt.scatter(Right['our.x'] , Right['our.y'] , s=.01 , c = 'b' , marker = '.' )

#Righties and Lefties together
plt.scatter(Left['our.x'] , Left['our.y'] , s=.01 , c='r' , marker='.')
plt.scatter(Right['our.x'] , Right['our.y'] , s=.01 , c = 'b' , marker = '.' )

#%% Assignment - Part 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Load the data containing the NBA shot log for the 2016/17 season that we used earlier
shot = pd.read_csv('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\NBA Shotlog_16_17.csv')

#Use the same code that we used to project all shots onto a half court by defining shot[‘halfcourt_x’] and shot[‘halfcourt_y’]
shot['halfcourt_x'] = np.where(shot['location_x'] < 933/2 , 933 - shot['location_x'] , shot['location_x']) #create a new variable where if location_x is less than 933/2, then return 933 - location_x, otherwise return the same value
shot['halfcourt_y'] = np.where(shot['location_y'] < 933/2 , 500 - shot['location_y'] , shot['location_y']) #create a new variable where if location_y is less than 933/2, then return 500 - location_y, otherwise return the same value
shot.describe()

#Now define subsets for the following players: Kevin Durant, Dwight Howard, DeAndre Jordan and Russell Westbrook.
Durant= shot[shot['shoot_player'] == 'Kevin Durant']
Howard = shot[shot['shoot_player'] == 'Dwight Howard']
Jordan = shot[shot['shoot_player'] == 'DeAndre Jordan']
Westbrook = shot[shot['shoot_player'] == 'Russell Westbrook']

Durant.describe()
Howard.describe()
Jordan.describe()
Westbrook.describe()
#Create plots of their shots in the same way that we did for Steph Curry and LeBron James: copy the code we used and just change the names. Show the plots of Russell Westbrook and Kevin Durant side by side. In order to make sure that the two plots have the same ranges, for each subplot add the lines:
hxD = Durant['halfcourt_x']
hyD= Durant['halfcourt_y']
colors = np.where(Durant['current_shot_outcome'] == 'SCORED' , 'r' , np.where(Durant['current_shot_outcome'] == 'MISSED' , 'b' , 'g')) 
plt.figure(figsize = (94/12 , 50/6))
plt.scatter(hxD , hyD , s = 10, c = colors , marker = '.')
plt.grid(True) #Turn on grid
plt.title('Durant' , fontsize = 15)
plt.xlim(500,950)
plt.ylim(0,500)

hxJ = Jordan['halfcourt_x']
hyJ= Jordan['halfcourt_y']
colors = np.where(Jordan['current_shot_outcome'] == 'SCORED' , 'r' , np.where(Jordan['current_shot_outcome'] == 'MISSED' , 'b' , 'g')) 
plt.figure(figsize = (94/12 , 50/6))
plt.scatter(hxJ, hyJ , s = 10, c = colors , marker = '.')
plt.grid(True) #Turn on grid
plt.title('DeAndre Jordan' , fontsize = 15)
plt.xlim(500,950)
plt.ylim(0,500)

hxH = Howard['halfcourt_x']
hyH= Howard['halfcourt_y']
colors = np.where(Howard['current_shot_outcome'] == 'SCORED' , 'r' , np.where(Howard['current_shot_outcome'] == 'MISSED' , 'b' , 'g')) 
plt.figure(figsize = (94/12 , 50/6))
plt.scatter(hxH,  hyH , s = 10, c = colors , marker = '.')
plt.grid(True) #Turn on grid
plt.title('Dwight Howard' , fontsize = 15)
plt.xlim(500,950)
plt.ylim(0,500)

hxW = Westbrook['halfcourt_x']
hyW= Westbrook['halfcourt_y']
colors = np.where(Westbrook['current_shot_outcome'] == 'SCORED' , 'r' , np.where(Westbrook['current_shot_outcome'] == 'MISSED' , 'b' , 'g')) 
plt.figure(figsize = (94/12 , 50/6))
plt.scatter(hxW,  hyW , s = 10, c = colors , marker = '.')
plt.grid(True) #Turn on grid
plt.title('Russell Westbrook' , fontsize = 15)
plt.xlim(500,950)
plt.ylim(0,500)

#Now we can even put the two player plots side-by-side (Westbrook and Durant)
f = plt.figure(figsize = (94/6 , 50/6)) 
ax = f.add_subplot(121) 
colors = np.where(Westbrook['current_shot_outcome'] == 'SCORED' , 'r' , np.where(Westbrook['current_shot_outcome'] == 'MISSED' , 'b' , 'g'))
ax = plt.scatter(hxW , hyW, s=10 , c = colors , marker = '.')
plt.grid(True)
plt.title('Russell Westbrook' , fontsize = 15)
plt.xlim(500,950)
plt.ylim(0,500)

ax = f.add_subplot(122) 
colors = np.where(Durant['current_shot_outcome'] == 'SCORED' , 'r' , np.where(Durant['current_shot_outcome'] == 'MISSED' , 'b' , 'g'))
ax = plt.scatter(hxD , hyD , s=10 , c = colors , marker = '.')
plt.grid(True)
plt.title('Kevin Durant' , fontsize = 15)
plt.xlim(500,950)
plt.ylim(0,500)

#Create the plot of DeAndre Jordan and Dwight Howard side by side
f = plt.figure(figsize = (94/6 , 50/6)) 
ax = f.add_subplot(121) 
colors = np.where(Jordan['current_shot_outcome'] == 'SCORED' , 'r' , np.where(Jordan['current_shot_outcome'] == 'MISSED' , 'b' , 'g'))
ax = plt.scatter(hxJ , hyJ, s=10 , c = colors , marker = '.')
plt.grid(True)
plt.title('DeAndre Jordan' , fontsize = 15)
plt.xlim(500,950)
plt.ylim(0,500)

ax = f.add_subplot(122) 
colors = np.where(Howard['current_shot_outcome'] == 'SCORED' , 'r' , np.where(Howard['current_shot_outcome'] == 'MISSED' , 'b' , 'g'))
ax = plt.scatter(hxH , hyH , s=10 , c = colors , marker = '.')
plt.grid(True)
plt.title('Dwight Howard' , fontsize = 15)
plt.xlim(500,950)
plt.ylim(0,500)

#Create the plot of Brook Lopez and Robin Lopez side by side
BLopez= shot[shot['shoot_player'] == 'Brook Lopez']
RLopez = shot[shot['shoot_player'] == 'Robin Lopez']

hxBL = BLopez['halfcourt_x']
hyBL= BLopez['halfcourt_y']
hxRL = RLopez['halfcourt_x']
hyRL= RLopez['halfcourt_y']

f = plt.figure(figsize = (94/6 , 50/6)) 
ax = f.add_subplot(121) 
colors = np.where(BLopez['current_shot_outcome'] == 'SCORED' , 'r' , np.where(BLopez['current_shot_outcome'] == 'MISSED' , 'b' , 'g'))
ax = plt.scatter(hxBL , hyBL, s=10 , c = colors , marker = '.')
plt.grid(True)
plt.title('Brook Lopez' , fontsize = 15)
plt.xlim(500,950)
plt.ylim(0,500)

ax = f.add_subplot(122) 
colors = np.where(RLopez['current_shot_outcome'] == 'SCORED' , 'r' , np.where(RLopez['current_shot_outcome'] == 'MISSED' , 'b' , 'g'))
ax = plt.scatter(hxRL , hyRL , s=10 , c = colors , marker = '.')
plt.grid(True)
plt.title('Robin Lopez' , fontsize = 15)
plt.xlim(500,950)
plt.ylim(0,500)

#%% Assignment - Part 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load the data for the IPL ball by ball (IPLbyb) and run the two functions that allow you to plot the game run/wicket curves for each game, and to plot multiple games alongside each other.
IPLbyb = pd.read_excel('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\IPLbyb.xlsx')

def plot_runs_wickets(IPLbyb, ax):
    gameno = IPLbyb['gameno'].unique()[0]
    for inning, data in IPLbyb.groupby('innings_number'):
        #create seprate dataframe for wickets
        wicket = data[data['wicket'] > 0]
        #plots line
        ax.plot(data['delivery_no'] , data['runs_total_end'])
        #plots markers
        marker = 'bo' if inning == 1 else 'ro'
        ax.plot(wicket['delivery_no'], wicket['runs_total_end'] , marker)
        #labels
        ax.set_xlabel('balls')
        ax.set_ylabel('runs')
        ax.set_title(f'Game{gameno}') #F-string to include game number variable
    ax.legend(['runs1' , 'wkt1' , 'runs2' , 'wkt2'])

#Second, a function that allows us to plot two or more games at the same time
def plot_runs_wickets_multigame(list_games):
    n = len(list_games)
    fig, axs = plt.subplots(n , 1, figsize = (6,15))
    for i , gameno in enumerate(list_games):
        game = IPLbyb[IPLbyb['gameno'] == gameno]
        plot_runs_wickets(game, axs[i] if n > 1 else axs)

#These two functions will allow us to produce multiple charts to display games alongside each other
#But first we should generate a list of games so that we can decide which game's number refer to which teams
#Identify if the home team batted first
IPLbyb['hometeambatsfirst'] = np.where((IPLbyb['home team'] == IPLbyb['batting_team']) & (IPLbyb['innings_number'] == 1) , 'yes' , 'no')

#Drop duplicates so we just have a list of games
games = IPLbyb.drop_duplicates('gameno')

#Generate list of games
games = games[['gameno' , 'home team' , 'batting_team' , 'bowling_team' , 'hometeambatsfirst']]

#Create a new column called road team by using the apply function to apply a 
#Lambda function across each row (axis = 1)
#Lambda function takes row 'x' (batting_team) as input, then checks if the value in 'home_team' is equal to the value.  
games['road team'] = games.apply(lambda x: x['batting_team'] if x['home team'] == x['bowling_team'] else x['bowling_team'], axis = 1)

#This line of code is an assert statement to verify a condition about games df
#We want to compare and make sure that no home team = away team 
assert (games['home team'] !=games['road team']).all()

#Reduce df now to just know which team bats first and gameno
games = games[['gameno' , 'home team' , 'road team' , 'hometeambatsfirst']]

#Now generate the plots for the pair of games between

#a) Kings XI Punjab and Delhi Daredevils
plot_runs_wickets_multigame([2,22])

#b) Kolkata Knight Riders and Royal Challengers Bangalore
plot_runs_wickets_multigame([3,29])

#c) Sunrisers and Rajasthan Royals
plot_runs_wickets_multigame([4,28])

#d) Chennai Super Kings and Kolkata Knight Riders
plot_runs_wickets_multigame([5,33])




























