#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# In[10]:


df = pd.read_csv('./Train.csv')
df
df_clean = df.drop(['name','ticket','embarked','body','home.dest','boat','cabin'],axis=1)
df_clean


# In[11]:


le = LabelEncoder()


# In[12]:


df_clean['sex'] = le.fit_transform(df_clean['sex']) 
df_clean


# In[13]:


df_clean = df_clean.fillna(df_clean['age'].mean())
df_clean


# In[15]:


df_clean.info()


# In[18]:


X = df_clean[['pclass','sex','age','sibsp','parch','fare']]
X


# In[19]:


Y = df_clean[['survived']]
Y


# In[20]:


from sklearn.tree import DecisionTreeClassifier


# In[21]:


dt = DecisionTreeClassifier(criterion='entropy',max_depth=5)


# In[22]:


dt.fit(X,Y)


# In[30]:


df_test = pd.read_csv('./Test.csv')
df_test = df_test.drop(['name','ticket','embarked','body','home.dest','boat','cabin'],axis=1)
df_test['sex'] = le.fit_transform(df_test['sex']) 
df_test = df_test.fillna(df_test['age'].mean())


# In[31]:


df_test


# In[33]:


Y_test = dt.predict(df_test)


# In[34]:


Y_test


# In[35]:


dt.score(X,Y)


# In[38]:


dfY = pd.DataFrame(Y_test)
dfY.to_csv('ans.csv')


# In[ ]:




