
# coding: utf-8

# In[3]:


# Anything start with '#' is comment just to explain what code is doing
# import libraries to use inbuilt functions
import pandas as pd
import numpy as np
import os
#import scikit-learn as sk

# get current working directory
print(os.getcwd())
# change your working directory to where your data is saved
os.chdir('C:\Users\MukeshMadesia\Downloads\Data')
print(os.getcwd())
# to change to titanic folder -- could be done above only
# './' given current path
os.chdir('./titanic')
# import data from file
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# to display few rows of data imported above; not required
s = df_train.head()
print(s)
t = df_test.head()
print(t)
df_train.corr()


# In[10]:


# Sex is categorical data cannot be used in mathematical function 
# we will have to convert Sex into mathematical binary data - like 0 - female ; 1- male
# Below syntax will add new column in table with heading "male" and value 0 for female & 1 for male
df_train["male"] = np.where(df_train.Sex == 'male',1,0)
df_train.head()


# In[17]:


# create seperate dataframe (table) for independent varibale
# Independent Varibale - are those using which we will predict the outcome - here Pclass & Sex of passenger
# Dependent Variable  - is that which we have to predict - here Survival of passanger

df_independent_variable = pd.DataFrame(df_train[['Pclass','male']])
df_independent_variable.head()


# In[18]:


df_dependent_variable = pd.DataFrame(df_train[['Survived']])
df_dependent_variable.head()


# In[19]:


# function required to split the data in two parts - Training & Test Data
# Train data - will be used to create Model(mathmatical relation/function) 
# Test data - will be used to validate our suggested model
# Note - Not to be confused with - Test data given in Challange - thats final test data for challange as whole, we dont have 
# actual outcome of it 
from sklearn.model_selection import train_test_split as tts


# In[21]:


df_independent_variable_train,df_independent_variable_test,df_dependent_variable_train,df_dependent_variable_test = tts(df_independent_variable,df_dependent_variable,random_state=0)


# In[27]:


# This is to import inbuilt Logistic Regression function
# why this one directly - we are doing hit and trail, and it requires theoretical understanding of different ML function
# to understand which one may fit better in given situation
from sklearn.linear_model import LogisticRegression


# In[29]:


# This is to just shorten the function name for convenience further use
lr = LogisticRegression()

lr.fit(df_independent_variable_train,df_dependent_variable_train)


# In[30]:


# this is to check the accuracy of our model - 
lr.score(df_independent_variable_test,df_dependent_variable_test)


# In[32]:


# To Predict the survival (result)
lr.predict(df_independent_variable_test)


# In[ ]:


''' 
This is additional information, 
DataFrame - is data structure in Python, two-dimensional , tabular format
'''

