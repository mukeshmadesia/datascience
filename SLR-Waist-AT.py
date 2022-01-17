# -*- coding: utf-8 -*-
"""
Spyder Editor

Linear Regression - Waist Circumference vs Adipose Tissue
"""

#Check currect working directory 

import os
import pandas as pd

import matplotlib.pyplot as plt
#matplotlib inline

import statsmodels.formula.api as smf

import numpy as np

os.curdir
os.chdir('C:\\Users\\Admin\\Desktop\\ExcelR\\Data')

data = pd.read_csv('wc-at.csv')
print('Sample data:\n',data.head())

print('Data infor:\n',data.info())

print('**Data Description\*\n',data.describe())

Waist_In = pd.DataFrame(data['Waist'])

print('*************Model 1 - AT vs Waist********')

model = smf.ols('AT~Waist', data)

model_fit = model.fit()
print('R-square:\n', model_fit.rsquared)
print('R-squared-adjusted:\n',model_fit.rsquared_adj)

print('Coefficient:\n',model_fit.params)



plt.Figure(figsize=(5,5))
plt.boxplot(data.Waist)
plt.show()
plt.plot(data.Waist,data.AT,"Blue" )

plt.plot(data.Waist,data.AT,"bo" )

print('Model Summary**\n',model_fit.summary())
#print('Model Residual:\n',model_fit.resid)
#print('Model Pearson:\n',model_fit.resid_pearson)



pred1 = model_fit.predict(Waist_In)
plt.show()
plt.scatter(x=data['Waist'],y=data['AT'],color='black')
plt.show()

rmse_linear = np.sqrt(np.mean((np.array(data['AT'])-np.array(pred1))**2))



print('*********************Model 2 AT vs Log(Waist)***********')
model2 = smf.ols('AT~np.log(Waist)', data).fit()

#model_fit = model.fit()
print('R-square:\n', model2.rsquared)
print('R-squared-adjusted:\n',model2.rsquared_adj)

print('Coefficient:\n',model2.params)



plt.Figure(figsize=(5,5))
plt.boxplot(data.Waist)
plt.show()
plt.plot(data.Waist,data.AT,"Blue" )

plt.plot(data.Waist,data.AT,"bo" )

print('Model Summary**\n',model2.summary())
#print('Model Residual:\n',model_fit.resid)
#print('Model Pearson:\n',model_fit.resid_pearson)


AT_pred=model2.predict(Waist_In)
plt.show()
plt.scatter(x=data['Waist'],y=data['AT'],color='black')
plt.show()

pred2 = model2.predict(Waist_In)

rmse_log = np.sqrt(np.mean((np.array(data['AT'])-np.array(pred2))**2))


print('*********************Model 3 log(AT) vs (Waist)***********')
model3 = smf.ols('np.log(AT)~Waist', data).fit()

#model_fit = model.fit()
print('R-square:\n', model3.rsquared)
print('R-squared-adjusted:\n',model3.rsquared_adj)

print('Coefficient:\n',model3.params)



plt.Figure(figsize=(5,5))
plt.boxplot(data.Waist)
plt.show()
#plt.plot(data.Waist,data.AT,"Blue" )

#plt.plot(data.Waist,data.AT,"bo" )

print('Model Summary**\n',model3.summary())
#print('Model Residual:\n',model_fit.resid)
#print('Model Pearson:\n',model_fit.resid_pearson)


pred3 = np.exp(model3.predict(Waist_In))

rmse_log_y = np.sqrt(np.mean((np.array(data['AT'])-np.array(pred3))**2))

print('RMSE Linear:',rmse_linear)
print('RMSE log:',rmse_log)
print('RMSE log-y:',rmse_log_y)

print(pred2)
print(pred3)

plt.scatter(x=data['Waist'],y=data['AT'],color='green')
plt.plot(data.Waist,pred1,color='red')
plt.plot(data.Waist,pred2,color='black')
plt.plot(data.Waist,pred3,color='blue')
