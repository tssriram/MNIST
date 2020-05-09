#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.neural_network import MLPClassifier


# In[2]:

# Dataset from Kaggle MNIST
train=pd.read_csv("C:/Users/Tarun/Desktop/New folder/train.csv")


# In[3]:

# Peruse the data
train.head()


# In[4]:

# Standardizing the pixel intensity values from 0-1
train.iloc[:,1:785]=train.iloc[:,1:785]/255


# In[5]:


# Defining the classifier object

mlp = MLPClassifier(hidden_layer_sizes=(196,49,49),activation='relu',solver='adam',max_iter=400)
# With this setup accuracy tested on Kaggle was 0.977

#mlp = MLPClassifier(hidden_layer_sizes=(49,49,49),activation='relu',solver='adam',max_iter=300)
# With this setup accuracy tested on Kaggle was 0.96814

#mlp = MLPClassifier(hidden_layer_sizes=(28,28,28,28),activation='relu',solver='adam',max_iter=250)
# With this setup accuracy tested on Kaggle was 0.96042

#mlp = MLPClassifier(hidden_layer_sizes=(28,28,28,28),activation='relu',solver='sgd')
# With this setup accuracy tested on Kaggle was 0.95514

#mlp = MLPClassifier(hidden_layer_sizes=(28,28,28,28),activation='relu',solver='sgd',max_iter=1000)
# With this setup accuracy tested on Kaggle was 0.954

#mlp = MLPClassifier(hidden_layer_sizes=(28,28,28,28),activation='relu',solver='lbfgs',max_iter=400)
# With this setup accuracy tested on Kaggle was 0.95214

# In[6]:

# Training the model
mlp.fit(train.iloc[:,1:785],train.iloc[:,0])


# In[7]:

#Reading in the test dataset from Kaggle MNIST
test=pd.read_csv("C:/Users/Tarun/Desktop/New folder/test.csv")


# In[8]:

# Standardizing it for making predictions
test=test/255


# In[9]:


print(test)


# In[10]:


pred=mlp.predict(test)


# In[11]:


print(pred)


# In[19]:

# Regular format for submission
a=pd.Series(range(1,28001))
predi={'ImageId':a,'Label':pred}
final=pd.DataFrame(predi)


# In[20]:


print(final)


# In[22]:

#Output into a file
final.to_csv('finalmnistmlp.csv', index=False)






