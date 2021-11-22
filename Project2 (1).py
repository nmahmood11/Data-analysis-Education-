#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import required libararies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('whitegrid')


# In[4]:


#Import data file in CSV formate
df=pd.read_csv(r'C:\Users\Noman Mahmood\Desktop\DS\University ranking Dataset\cwurData.csv')


# In[5]:


df.head()


# In[6]:


# Expalore data type and null values
df.info()


# In[7]:


# Collect basic statistics about data
df.describe().T


# In[8]:


#column names
df.columns


# In[9]:


df.isnull().sum()


# In[10]:


df['broad_impact'].nunique()


# In[11]:


#Data visualization
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Graph", fontweight="bold")


# In[12]:


df.cov()


# In[13]:


#Biverse Data analysis
sns.barplot(x=df['world_rank'].head(5), y=df['publications'])


# In[14]:


df['country'].unique()


# In[15]:


df['country'].value_counts()


# In[16]:


#deep analysis
sns.pairplot(df)


# In[17]:


df.drop('broad_impact', axis=1, inplace=True)


# In[18]:


#Import more libararies
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[19]:


df.isnull().sum()


# In[20]:


encoder = LabelEncoder()

df['country'] = encoder.fit_transform(df['country'])
country_mappings = {index: label for index, label in enumerate(encoder.classes_)}


# In[21]:


df.drop('institution', axis=1, inplace=True)
df.drop('year', axis=1, inplace=True)


# In[22]:


y = df['world_rank']
X = df.drop('world_rank', axis=1)


# In[23]:


scaler = StandardScaler()

X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=101)


# In[25]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[26]:


print(f"Model R^2: {model.score(X_test, y_test)}")


# In[ ]:




