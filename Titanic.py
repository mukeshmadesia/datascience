
# coding: utf-8

# In[2]:


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

