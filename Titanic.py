#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as pyplot
import pandas as pd


# In[4]:


df = pd.read_csv("desktop/tested.csv")
df.head(10)


# In[5]:


df.shape


# In[6]:


df.describe()


# In[8]:


df['Survived'].value_counts()


# In[11]:


sns.countplot(x=df['Survived'] , hue=df['Pclass'])


# In[13]:


df["Sex"]


# In[15]:


sns.countplot(x=df['Sex'],hue=df['Survived'])


# In[16]:


df.groupby('Sex')[['Survived']].mean()


# In[17]:


df['Sex'].unique()


# In[18]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
df['Sex']=labelencoder.fit_transform(df['Sex'])
df.head()


# In[19]:


df['Sex'], df['Survived']


# In[20]:


sns.countplot(x=df['Sex'],hue=df['Survived'])


# In[21]:


df.isna().sum()


# In[22]:


df=df.drop(['Age'],axis=1)


# In[23]:


df_final=df
df_final.head(10)


# In[27]:


X=df[['Pclass','Sex']]
Y=df['Survived']


# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[29]:


from sklearn.linear_model import LogisticRegression
log=LogisticRegression(random_state=0)
log.fit(X_train, Y_train)


# In[30]:


pred= print(log.predict(X_test))


# In[31]:


print(Y_test)


# In[32]:


import warnings
warnings.filterwarnings('ignore')
res=log.predict([[2,0]])
if(res==0):
    print("Not Survived")
else:
    print('survived')


# In[ ]:




