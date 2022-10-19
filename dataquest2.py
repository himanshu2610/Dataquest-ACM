#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:





# In[31]:


train_df=pd.read_csv('Dataquest_FE_SE_R2_train.csv')


# In[32]:


test_df=pd.read_csv('Dataquest_FE_SE_R2_test.csv')


# In[33]:


train_df.isnull().sum()


# In[34]:


test_df.isnull().sum()


# In[35]:


test_df['brightness_t21']=test_df['brightness_t21'].fillna(test_df['brightness_t21'].mean())


# In[36]:


test_df['scan']=test_df['scan'].fillna(test_df['scan'].mean())


# In[37]:


train_df.columns


# In[38]:


train_df['latitude'].dtype


# In[39]:


unique_val_region=[]
for i in train_df['region']:
    if(i not in unique_val_region):
        unique_val_region.append(i)
print(unique_val_region)


# In[40]:


unique_val_day=[]
for i in train_df['day/night']:
    if(i not in unique_val_day):
        unique_val_day.append(i)
print(unique_val_day)


# In[ ]:


# use one-hot encoding on these two variables


# In[41]:


y=train_df.explosion_intensity


# In[42]:


train_df.dropna(axis=0, subset=['explosion_intensity'], inplace=True)


# In[43]:


train_df.drop(['explosion_intensity'],axis=1,inplace=True)


# In[44]:


train_df['brightness_t31']=train_df['brightness_t31'].fillna(train_df['brightness_t31'].mean())
train_df['scan']=train_df['scan'].fillna(train_df['scan'].mean())
train_df['frp']=train_df['frp'].fillna(train_df['frp'].mean())
train_df.drop(['date'],axis=1,inplace=True)
test_df.drop(['date'],axis=1,inplace=True)


# In[45]:


train_df.info()


# In[46]:


# Get list of categorical variables
s = (train_df.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)


# In[47]:


# numerical columns
numerical_cols = [cname for cname in train_df.columns if train_df[cname].dtype in ['int64', 'float64']]
numerical_cols


# In[48]:


from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_df[object_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(test_df[object_cols]))


# In[49]:


OH_cols_train.index = train_df.index
OH_cols_test.index = test_df.index


# In[50]:


num_X_train = train_df.drop(object_cols, axis=1)
num_X_test = test_df.drop(object_cols, axis=1)


# In[51]:


OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)


# In[52]:


OH_X_train.drop(['id'],axis=1,inplace=True)


# In[54]:


OH_X_test.drop(['id'],axis=1,inplace=True)


# In[55]:


OH_X_train.head()


# In[57]:


OH_X_test.head()


# In[58]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=1)
model.fit(OH_X_train, y)
prediction=model.predict(OH_X_test)
output=pd.DataFrame({'id':test_df.id,'explosion_intensity':prediction})
output.to_csv('round_2.csv',index=False)

