
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')
import seaborn as sns


# # 1. Exploratory Data Analysis

# In[ ]:

train = pd.read_csv('train.csv', sep=',')
#test = pd.read_csv('test.csv',sep=',')


# In[ ]:

print("size of the data:",train.shape)


# In[ ]:

train.head(n=7)


# In[ ]:

# converting date_time to type (datetime) instead of string 
#train['date_time']=pd.to_datetime(train['date_time'],infer_datetime_format=True)
#train['srch_ci']=pd.to_datetime(train['srch_ci'],infer_datetime_format=True)
#train['srch_co']=pd.to_datetime(train['srch_co'],infer_datetime_format=True)

#train.dtypes


# In[ ]:

train.columns


# In[ ]:

hotel_cluster = train['hotel_cluster'].unique()
np.sort(hotel_cluster)


# In[ ]:

train['hotel_cluster'].describe()


# In[ ]:

for columns in train:
    print ('Column Name: ',columns)
    print (train[columns].unique())


# ### Managing the Missing Data 

# In[ ]:

# Number of missing data 
train.isnull().sum(axis=0)


# In[ ]:

train_2 = train.dropna()
print ("size of the data without the missing data:", train_2.shape)


# ### Converting Check In date, and Check Out date

# In[ ]:

train_2['ci_year']=train_2['srch_ci'].apply(lambda x: x[:4]).astype(int)
train_2['co_year']=train_2['srch_co'].apply(lambda x: x[:4]).astype(int)


# In[ ]:

train2=train_2[train_2['ci_year']<2020]
train2['co_year'].unique()


# In[ ]:

train2['ci_year'].unique()


# In[ ]:

train2['srch_ci']=pd.to_datetime(train2['srch_ci'],infer_datetime_format=True)
train2['srch_co']=pd.to_datetime(train2['srch_co'],infer_datetime_format=True)


# In[ ]:

train2['stay']= train2['srch_co']- train2['srch_ci']


# In[ ]:

train2['stay']=train2['stay'].apply(lambda x: x.days)


# In[ ]:

train3 = train2.drop(['date_time','srch_ci','srch_co'],axis=1)
train3.dtypes


# # 2. Splitting the data into Training data and Testing data 

# In[ ]:

from sklearn.model_selection import train_test_split
train, test = train_test_split(train3, test_size=0.2)


# In[ ]:

train.shape


# # 3. Unsupervised K-Means Clustering

# In[ ]:

from sklearn.cluster import KMeans


# In[ ]:

mat = train.as_matrix()


# In[ ]:

kmeans = KMeans(n_clusters=5,random_state=0).fit(mat)
kmeans.labels_


# In[ ]:

kmeans.cluster_centers_

