# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 10:13:50 2022

@author: Admin
"""

import os
import pandas as pd
import seaborn as sns

import warnings

warnings.filterwarnings(action='ignore')

os.chdir('C:\\Users\\Admin\\Desktop\\ExcelR\\Data')
data = pd.read_csv('NewspaperData.csv')
print("Data sample",data.head())

print("Printing Info:", data.info())
print("Printing corr details:", data.corr())

## Exploratory Analysis

sns.distplot(data['daily'])
sns.distplot(data['sunday'],hist=False)
sns.distplot(data['sunday'],kde=False)



import statsmodels.formula.api as smf
model = smf.ols("sunday~daily",data = data).fit()

sns.regplot(x='daily',y='sunday',data=data)
print('Coefficiants\n',model.params)

print('model t values\n',model.tvalues)
print('model p values\n',model.pvalues)

print("R Squared value\n", model.rsquared ,'\nR Square Adjusted\n' ,model.rsquared_adj)

### predict new data

newdata = pd.Series([250,350])
df_newdata = pd.DataFrame(newdata,columns=['daily'])

print('Predicted valuse \n', model.predict(df_newdata))
      
    
      