
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')
import seaborn as sns


# In[2]:

train = pd.read_csv('train.csv', sep=',')
#test = pd.read_csv('test.csv',sep=',')


# In[3]:

print("size of the data:",train.shape)


# In[32]:

train_2 = train.dropna()
print ("size of the data without the missing data:", train_2.shape)


# In[3]:

train.columns


# In[4]:

train.head(n=7)


# In[8]:

hotel_cluster = train['hotel_cluster'].unique()
np.sort(hotel_cluster)


# In[11]:

train['hotel_cluster'].describe()


# In[6]:

# types of the data
train.dtypes


# In[25]:

# converting date_time to type (datetime) instead of string 
train['date_time']=pd.to_datetime(train['date_time'],infer_datetime_format=True)
#train['srch_ci']=pd.to_datetime(train['srch_ci'],infer_datetime_format=True)
#train['srch_co']=pd.to_datetime(train['srch_co'],infer_datetime_format=True)

train.dtypes


# In[31]:

from datetime import datetime
for k in range (len(train['srch_ci'])):
    train['srch_ci']= datetime.strptime(train['srch_ci'].iloc[k], '%Y-%m-%d')


# ## Converting Check In date, and Check Out date

# In[36]:

train_2['ci_year']=train_2['srch_ci'].apply(lambda x: x.year).astype(int)
train_2['ci_year'].head()


# In[37]:

train_2[['ci_year']<2020]


# In[8]:

train.describe()


# In[16]:

train['num of stay']= train['srch_co']- train['srch_ci']
train['num of stay'].head()


# In[9]:

# Number of unique users
train['user_id'].unique().size


# In[24]:

for columns in train:
    print ('Column Name: ',columns)
    print (train[columns].unique())


# In[10]:

# Number of missing data 
train.isnull().sum(axis=0)


# In[11]:

train['hotel_cluster'].unique().size


# In[13]:

train['month']=train['date_time'].apply(lambda x: x.month).astype(str)


# In[14]:

train['month'].head()


# In[ ]:

sns.counterplot('booki')


# ### Unsupervised Clustering

# In[12]:

from sklearn.cluster import KMeans


# In[13]:

mat = train.as_matrix()
mat


# In[ ]:

trial1 = train[]


# In[16]:

kmeans = KMeans(n_clusters=5,random_state=0).fit(mat)
kmeans.labels_


# In[ ]:

kmeans.cluster_centers_


# In[ ]:



