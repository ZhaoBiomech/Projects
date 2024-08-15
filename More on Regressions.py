# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 08:11:55 2024

@author: William Zhao
"""
#%% The Salary-Performance Relationship in the English Premier League
#In this module we will examine the impact of salary (relative to the average of the season)
#on team performance (measured by league position). We will also examine how
#the addition of potential omitted vars impact estimates

import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

#Import the EPL data
EPL = pd.read_excel('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\EPL pay and performance.xlsx')
EPL.describe()
EPL.info()

#First, we will use the .groupby function to sum salaries to get a total league salary of each season
Sumsal = EPL.groupby (['Season_ending'])['salaries'].sum().reset_index().rename(columns = {'salaries' : 'allsal'})
Sumsal
#If we prefer to see the numbers in long format rather than scientific notation
pd.options.display.float_format = '{:.0f}'.format
Sumsal

#We want to compare team spending relative to the avg of that season so we will
#merge the aggreagate salaries back to the main df
EPL = pd.merge(EPL , Sumsal , on = ['Season_ending'] , how = 'left')

#Now we create a var for the relative salary in the EPL
EPL['relsal'] = EPL['salaries'] / EPL['allsal']

#Before running a regression, lets plot the relationship between salary and wpc
sns.regplot(x = 'relsal' , y = 'Position' , data = EPL , ci = False)
#The chart shows that there is a neg relationship, but that is because a lower numerical position
#means better performance and a higher position in the league
#To avoid confusion, we can reverse the relationship by flipping the sign of Position
EPL['mpos'] = -EPL['Position']
sns.regplot(x = 'relsal' , y = 'mpos' , data = EPL , ci = False)

#One thing we notice is that there is some curvature to the data. This is a common feature 
#of many types of data. We estimate a linear relationship, so it would be better to first linearize
#the data by taking the logarithm of the data.
EPL['lnpos'] = -np.log(EPL['Position'])
sns.regplot(x = 'relsal' , y = 'lnpos' , data = EPL , ci = False)

#We can now run the simple regression of league position on salaries
reg_EPL1 = sm.ols(formula = 'lnpos ~ relsal' , data = EPL).fit()
print(reg_EPL1.summary()) # R^2 = 0.657, relsal coeff = 23.9

#Although relsal seems to be quite a strong predictor of performance in the EPL
# we now consider other factors to see if omitted variable bias is present in our model
#The first factor we will consider is specific to the promotion/relegation system
reg_EPL2 = sm.ols(formula = 'lnpos ~ relsal + promoted_last_season' , data = EPL).fit()
print(reg_EPL2.summary()) #promotion is statistically insignficant, so we can drop it

#We now consider the impact of lagged dependent variable - league position in the previous season
EPL.sort_values(by = ['Club' , 'Season_ending'] , ascending = True) #First sort the df by teams and by season

EPL['lnpos_lag'] = EPL.groupby('Club')['lnpos'].shift(1) # Group by Club and use the .shift(1) command to create the lag of league position
EPL

#Now we run a third regression
reg_EPL3 = sm.ols(formula = 'lnpos ~ lnpos_lag + relsal' , data = EPL).fit()
print(reg_EPL3.summary()) # R^2 = 0.708 , adding lnpos_lag decreased the relsal coefficient, implying that the omission of the lag dependent var inflated our estimate of relsal

#Finally, we will consider the possible effects of heterogeneity by adding fixed effects
# into our regression. 

reg_EPL4 = sm.ols(formula = 'lnpos ~ lnpos_lag + relsal + C(Club)' , data = EPL).fit()
print(reg_EPL4.summary()) # R^2 = 0.746 , relsal decreased even more

#However, the above regression is flawed because when estimated fixed effects, there must always be a reference group.
#In the above case, it used the first team in the list, which is Arsenal, a historically well performing team.
#It would be more appropriate to evaluate fixed effects using an average mid-table team
Avpos = EPL.groupby(['Club'])['Position'].mean() #Find the avg position of each team
Avpos

#We can see from Avpos that Everton was a consistent mid-table team, so we will use them
#as your reference group
reg_EPL5 = sm.ols(formula = "lnpos ~ lnpos_lag + relsal + C(Club,Treatment('Everton'))" , data = EPL).fit() #We expand the C() statement to include a 'treatment' group reference
print(reg_EPL5.summary())
#We now see that only four clubs have statistically significant coefficients. 
#Two of these are Manchester United and Arsenal, the two dominant clubs over the period. 
#This implies that these clubs, which spent more money than the others on players, 
#still managed to extract better than average performance from these players.


#Now we can consider how changes in relsal affect league position
# Ignoring the fixed effects and the lagged dependent variable, 
#minus the log of league position can be expressed as a function of the constant 
#plus the relsal coefficient times the value of relsal, i.e. -lnpos = -2.1 + 11 relsal.

#Let's consider three relsal values : .02 , .07 and .14
#To convert -lnpos back into position we have to multiply by -1 and then take the exponent.
#To take an exponent using numpy you just type np.exp() with the expression in parentheses.
print(np.exp(2.1 - 11 * 0.02)) #6.55
print(np.exp(2.1 - 11 * 0.08)) #3.39
print(np.exp(2.1 - 11 * 0.14)) #1.75

#%% Explaining Relationships using Regression Analyses - NBA
#In this module we will do what we did with the EPL data, but now with NBA data

import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

#Import data
NBA = pd.read_excel('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\NBA pay and performance.xlsx')
NBA

#First we create a sum of league salary to see the total spending over each season
Sumsal = NBA.groupby(['season'])['salaries'].sum().reset_index().rename(columns={'salaries' : 'allsal'})
Sumsal #We can see increasing total salary from 2012 -> 2018

#Now merge Sumsal back into the NBA df and calculate a relative salary var
NBA = pd.merge(NBA , Sumsal , on = ['season'] , how = 'left')
NBA['relsal'] = NBA['salaries'] / NBA['allsal']

#Before running a regression, lets plot the data to see what it looks like
sns.regplot(x = 'relsal' , y = 'wpc' , data = NBA , ci = False) #Clear position relationship between relsal and wpc

#Lets run our first regression now
reg_NBA1 = sm.ols(formula = 'wpc ~ relsal' , data = NBA).fit()
print(reg_NBA1.summary()) #R^2 = 0.172 , relsal coeff = 11.3

#While salary  should capture many aspects of team quality, salaries are not renegotiated every year, 
#and many aspects of team quality would be in place in the previous season. 
#So we can add a lagged dependent variable that captures the wpc of a team in the previous season 
#to our regression, and then see if this changes our estimate of the impact of salaries.
NBA.sort_values(by = ['Team' , 'season'] , ascending = True)
NBA['wpc_lag'] = NBA.groupby('Team')['wpc'].shift(1)

#We can now run our 2nc regression to include our lag var
reg_NBA2 = sm.ols(formula = 'wpc ~ relsal + wpc_lag' , data = NBA).fit()
print(reg_NBA2.summary()) #R^2 = 0.416 and relsal is no longer statistically sig!

#Clearly, not all teams are identical, while our regression treats each team as if they were, 
#In our regression specification we want to find balance between treating each team as if it were identical, 
#and treating each team as if it were completely unique. The truth is likely to be that there are common factors affecting all teams, 
#but that there are also idiosyncrasies. This is often described as heterogeneity.
#One way we can introduce heterogeneity is through fixed effects. Estimation of fixed effects allows us to identify differences between 
#the teams that are independent of the impact of salaries or of the lagged dependent variable.
reg_NBA3 = sm.ols(formula = 'wpc ~ relsal + wpc_lag + C(Team)' , data = NBA).fit()
print(reg_NBA3.summary())
#A positive fixed effect means that in some way the team was able to perform above average, 
#and a negative fixed effect implies below average performance.

#%% Salaries and Performance in the NHL
#By looking at third league modeled on the North American system we can get a 
#better understanding of the three variables we have used to explain 
#win perecentage: salaries, lagged win percentage, and fixed effects.

import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

NHL = pd.read_excel('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\NHL pay and performance.xlsx')
NHL

#Sum all salaries each season to create a Sumsal variable
Sumsal = NHL.groupby(['season'])['salaries'].sum().reset_index().rename(columns = {'salaries' : 'allsal'})
Sumsal

#Merge the dfs and calculate a relsal var
NHL = pd.merge(Sumsal , NHL , on = ['season'] , how = 'left')
NHL['relsal'] = NHL['salaries'] / NHL['allsal']

#Plot relsal vs wpc
sns.regplot(x = 'relsal' , y='wpc' , data = NHL) #We see some overlap of the points so lets reduce their size
sns.regplot(x = 'relsal' , y='wpc' , data = NHL , scatter_kws={'s':5}) #scatter_kws={'s':5} to reduce marker size

#First regression
reg_NHL1 = sm.ols(formula = 'wpc ~ relsal' , data = NHL).fit()
print(reg_NHL1.summary())

#Now lets create a lag dependent var
NHL.sort_values(by = ['Team' , 'season'] , ascending = True)
NHL['wpc_lag'] = NHL.groupby('Team')['wpc'].shift(1)

#We can now run our 2nd regression that includes the lag dependent var
reg_NHL2 = sm.ols(formula = 'wpc ~ relsal + wpc_lag' , data = NHL).fit()
print(reg_NHL2.summary())

#Lets also add a fixed effect of team
reg_NHL3 = sm.ols(formula = 'wpc ~ relsal + wpc_lag + C(Team)' , data = NHL).fit()
print(reg_NHL3.summary())

#From our 3rd regression, we see the wpc_lag has lost significance, so let's drop it
reg_NHL4 = sm.ols(formula = 'wpc ~ relsal + C(Team)' , data = NHL).fit()
print(reg_NHL4.summary())

#This model has a fairly straightforward interpretation:
    # wpc = 0.256 + 8.76 * relsal + fixed effect
#If we ignore fixed effects, we can identify the expected win percentage for 
#low, average and high relative spending:
print(0.256 + 8.76*0.02) #0.43
print(0.256 + 8.76*0.0325) #0.54
print(0.256 + 8.76*0.045) #0.65

#Self-Test: Based on the fixed effects regression, calculate the win percentage of:
#(a) The Calgary Flames assuming the value of relsal for the team is 0.03 
print(0.256 + (8.76 * 0.03) - 0.08 ) #0.439

#(b) The Edmonton Oilers assuming the value of relsal for the team is 0.04 
print(0.256 + (8.76 * 0.04) - 0.14 ) #0.466

#(c) The Montreal Canadiens assuming the value of relsal for the team is 0.05
print(0.256 + (8.76 * 0.05) - 0.06 ) #0.634

#%% Salaries and Performance in MLB
#The final sport we will examine is baseball

import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

MLB = pd.read_excel('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\MLB pay and performance.xlsx')
MLB

#Create a Sumsal var
Sumsal = MLB.groupby(['season'])['salaries'].sum().reset_index().rename(columns = {'salaries' : 'allsal'})
Sumsal

#Merge the dfs and calculate relsal
MLB = pd.merge(MLB , Sumsal , on = ['season'] , how = 'left')
MLB['relsal'] = MLB['salaries'] / MLB['allsal']

#Plot relsal vs wpc
sns.regplot(x = 'relsal' , y = 'wpc' , data = MLB , scatter_kws = {'s' : 3})

#Run our first regression
reg_MLB1 = sm.ols(formula = 'wpc ~ relsal' , data = MLB).fit()
print(reg_MLB1.summary()) 
#As with the NBA, we find that the coefficient on relsal is highly significant, 
#but the size of our initial estimate is much smaller- recall that for the NBA the value was 11.3

#Based on this model, the wpc of a team with relsal = 4%
print(0.4301 + (2.0021 * 0.04)) #0.51 or 51%

#Lets create our lag dependent var
MLB.sortby_values(by = ['Team' , 'season'] , ascending = True)
MLB['wpc_lag'] = MLB.groupby('Team')['wpc'].shift(1)

#Now lets run our 2nd regression
reg_MLB2 = sm.ols(formula = 'wpc ~ relsal + wpc_lag' , data = MLB).fit()
print(reg_MLB2.summary())

#Self-Test: The model implies that win percentage of a team in year t, wpc(t) = 0.2839 +0.3614 x wpc_lag + 1.0259 x relsal
#Suppose relsal is 4% (0.04), calculate the value of wpc(t) if wpc(t-1) equals (a) 0.6 and (b) 0.4.
print(0.2839 + (0.3614 * 0.6) + (1.02590 * 0.04))
print(0.2839 + (0.3614 * 0.4) + (1.02590 * 0.04))

#Now lets run our regression to take into account fixed effects
reg_MLB3 = sm.ols(formula = 'wpc ~ relsal + wpc_lag + C(Team)' , data = MLB).fit()
print(reg_MLB3.summary())
#We can see that the adjusted_R has fallen from 0.233 -> 0.232, suggesting that the addition 
#of our fixed effects did not improve our model and that they should be ignored

#Therefore, Our preferred regression model is wpc(t) = 0.284 + 0.361 x wpc(t-1) + 1.026 x relsal (t)
#o work out the impact of relsal we need to eliminate the the lagged dependent variable from the equation, 
#which we do by assuming a "steady state"- where wpc(t) = wpc(t-1).
#wpc = 1/(1-0.361) x (0.284 + 1.026 x relsal)
print(1/(1-0.361)*(0.284 + 1.026*.01)) # 0.461
print(1/(1-0.361)*(0.284 + 1.026*.035)) # 0.501
print(1/(1-0.361)*(0.284 + 1.026*.06)) # 0.541

#%% Assignment 5 - Salary and Performance and the IPL
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

#Load the data
IPL = pd.read_excel('C:\\Users\\William Zhao\\Desktop\\Data Science Learning\\Data\\Files\\IPL (assignment) data.xlsx')
IPL
#Create the sum of salaries in each season
Sumsal = IPL.groupby('year')['salaries'].sum().reset_index().rename(columns = {'salaries' : 'allsal'})
Sumsal

IPL= pd.merge(IPL , Sumsal , on = ['year'] , how = 'left')

#Create a variable for team salary divided by total salaries for that season (relsal).
IPL['relsal'] = IPL['salaries'] / IPL['allsal']

#Create a value for win percentage. Define win percentage as wins divided games with a result (= games played minus games with no result). 
IPL['played_result'] = IPL['played'] - IPL['noresult']
IPL['wpc'] = IPL['won'] / IPL['played_result']

#Create the lagged value of win percentage for each team
IPL.sortby_values(by = ['team' , 'year'] , ascending = True)
IPL['wpc_lag'] = IPL.groupby('team')['wpc'].shift(1)

#Regress win percentage on:
#a) Relsal
reg_IPL1 = sm.ols(formula = 'wpc ~ relsal' , data = IPL).fit()
print(reg_IPL1.summary())

#b) Relsal + lagged win percentage
reg_IPL2 = sm.ols(formula = 'wpc ~ relsal + wpc_lag' , data = IPL).fit()
print(reg_IPL2.summary())

#c) Relsal + lagged win percentage  + team fixed effects
reg_IPL3 = sm.ols(formula = 'wpc ~ relsal + wpc_lag + C(team)' , data = IPL).fit()
print(reg_IPL3.summary())





















