#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.neural_network import MLPClassifier


# In[2]:


train=pd.read_csv("C:/Users/Tarun/Desktop/New folder/train.csv")


# In[3]:


train.head()


# In[4]:


train.iloc[:,1:785]=train.iloc[:,1:785]/255


# In[5]:



mlp = MLPClassifier(hidden_layer_sizes=(28,28,28,28),activation='relu',solver='sgd')


# In[6]:


mlp.fit(train.iloc[:,1:785],train.iloc[:,0])


# In[7]:


test=pd.read_csv("C:/Users/Tarun/Desktop/New folder/test.csv")


# In[8]:


test=test/255


# In[9]:


print(test)


# In[10]:


pred=mlp.predict(test)


# In[11]:


print(pred)


# In[19]:


a=pd.Series(range(0,28000))
predi={'ImageId':a,'Label':pred}
final=pd.DataFrame(predi)


# In[20]:


print(final)


# In[22]:


final.to_csv('finalmnistmlp.csv', index='False')






