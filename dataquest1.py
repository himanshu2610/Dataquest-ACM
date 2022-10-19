#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# In[4]:


df=pd.read_csv('train1.csv')


# In[5]:


test_df=pd.read_csv('test1.csv')


# In[6]:


test_df.head()


# In[7]:


test_df.isnull().sum().sort_values(ascending=False)


# In[8]:


df.shape


# In[9]:


df.isnull().sum().sort_values(ascending=False)


# In[10]:


df['fruit-shape'].value_counts()


# In[11]:


df['fruit-surface'].value_counts()


# In[12]:


df['bruises'].value_counts()


# In[13]:


df.info()


# In[14]:


df['stem-surface-above-peel']=df['stem-surface-above-peel'].fillna(df['stem-surface-above-peel'].mode()[0])


# In[15]:


df['stem-color-above-peel']=df['stem-color-above-peel'].fillna(df['stem-color-above-peel'].mode()[0])


# In[16]:


df['bruises']=df['bruises'].fillna(df['bruises'].mode()[0])


# In[17]:


df['fruit-surface']=df['fruit-surface'].fillna(df['fruit-surface'].mode()[0])


# In[18]:


df['stem-color-below-peel']=df['stem-color-below-peel'].fillna(df['stem-color-below-peel'].mode()[0])


# In[19]:


df['stem-shape']=df['stem-shape'].fillna(df['stem-shape'].mode()[0])


# In[20]:


df['survive']=df['survive'].fillna(df['survive'].mode()[0])


# In[21]:


round(df['stem-surface-below-peel'].mean())


# In[22]:


df['stem-surface-below-peel']=df['stem-surface-below-peel'].fillna(df['stem-surface-below-peel'].mean())


# In[23]:


df['fruit-odor']=df['fruit-odor'].fillna(df['fruit-odor'].mean())


# In[24]:


df['population']=df['population'].fillna(df['population'].mean())


# In[25]:


df['peel-quality']=df['peel-quality'].fillna(df['peel-quality'].mean())


# In[26]:


df['habitat']=df['habitat'].fillna(df['habitat'].mean())


# In[27]:


df['stem-root']=df['stem-root'].fillna(df['stem-root'].mean())


# In[28]:


df['peel-type']=df['peel-type'].fillna(df['peel-type'].mean())


# In[29]:


df['seed-size']=df['seed-size'].fillna(df['seed-size'].mean())


# In[30]:


df['seed-spacing']=df['seed-spacing'].fillna(df['seed-spacing'].mean())


# In[31]:


df['spores']=df['spores'].fillna(df['spores'].mean())


# In[32]:


df['fruit-shape']=df['fruit-shape'].fillna(df['fruit-shape'].mean())


# In[33]:


df['seed-color']=df['seed-color'].fillna(df['seed-color'].mean())


# In[34]:


df['fruit-color']=df['fruit-color'].fillna(df['fruit-color'].mean())


# In[35]:


test_df.isnull().sum().sort_values(ascending=False)


# In[36]:


test_df['stem-root']=test_df['stem-root'].fillna(test_df['stem-root'].mean())
test_df['seed-color']=test_df['seed-color'].fillna(test_df['seed-color'].mean())
test_df['population']=test_df['population'].fillna(test_df['population'].mean())
test_df['spores']=test_df['spores'].fillna(test_df['spores'].mean())
test_df['peel-quality']=test_df['peel-quality'].fillna(test_df['peel-quality'].mean())
test_df['peel-type']=test_df['peel-type'].fillna(test_df['peel-type'].mean())
test_df['peel-quality']=test_df['peel-quality'].fillna(test_df['peel-quality'].mean())
test_df['stem-surface-below-peel']=test_df['stem-surface-below-peel'].fillna(test_df['stem-surface-below-peel'].mean())
test_df['habitat']=test_df['habitat'].fillna(test_df['habitat'].mean())
test_df['fruit-shape']=test_df['fruit-shape'].fillna(test_df['fruit-shape'].mean())
test_df['seed-spacing']=test_df['seed-spacing'].fillna(test_df['seed-spacing'].mean())
test_df['seed-size']=test_df['seed-size'].fillna(test_df['seed-size'].mean())
test_df['fruit-color']=test_df['fruit-color'].fillna(test_df['fruit-color'].mean())
test_df['fruit-odor']=test_df['fruit-odor'].fillna(test_df['fruit-odor'].mean())


# In[37]:


test_df.isnull().sum()


# In[38]:


test_df.columns


# In[ ]:





# In[39]:


fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(df[['survive', 'fruit-shape', 'fruit-surface', 'fruit-color', 'bruises',
       'fruit-odor', 'seed-spacing', 'seed-size', 'seed-color', 'stem-shape',
       'stem-root', 'stem-surface-above-peel', 'stem-surface-below-peel',
       'stem-color-above-peel', 'stem-color-below-peel', 'peel-quality',
       'peel-type', 'spores', 'population', 'habitat'
       ]].corr(),annot=True,fmt='.2f',cmap='coolwarm')


# In[ ]:





# In[43]:


from sklearn.ensemble import RandomForestClassifier
y=df['survive']
features=['Id','fruit-shape', 'fruit-surface', 'fruit-color', 'seed-size','population', 'habitat']
model=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=1)
model.fit(df[features],y)
prediction=model.predict(test_df)
output=pd.DataFrame({'Id':test_df.Id,'survive':prediction})
output.to_csv('dataquest2.csv',index=False)

