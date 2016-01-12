
# coding: utf-8

# In[1]:

#Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')

# Set figure aesthetics
sns.set_style("darkgrid", {'ytick.major.size': 8.0})
sns.set_context("talk", font_scale=1.0)


# In[2]:

df = pd.read_csv('Responses.csv')
df.drop(df.columns[0],axis=1,inplace=True)


# In[3]:

df.head()


# In[4]:

#Split demographic data and responses
responses = df.drop(df.columns[0:20],axis=1)
demographs = df.drop(df.columns[20:],axis=1)


# In[5]:

#Number of Responses
print('Number of respondents:{0}'.format(responses.shape[0]))
print('Number of questions:{0}'.format(responses.shape[1]))
print('Total theoretical number questions of questions (users):{0}'.format(responses.size))

#Sparsity
print('\n%.2f%% of values are present' % (100*responses.notnull().sum().sum()/responses.size))
print('or equivalently,')
print('%.2f%% of values are missing\n' % (100*responses.isnull().sum().sum()/responses.size))

N = responses.shape[0]


# In[9]:

from scipy import sparse as ss
from scipy.sparse import csr_matrix


# In[7]:

responses.fillna(0,inplace=True)


# In[8]:

resp_ss = csr_matrix(responses.values)


# In[10]:

resp_ss = resp_spr


# In[ ]:

from sklearn.metrics.pairwise import pairwise_distances


# In[11]:

rxx = resp_ss.dot(resp_ss.T) #dot(X,X)


# In[37]:

N**2


# In[33]:

rxx


# In[15]:

rxx_diag = rxx.diagonal()


# In[30]:




# In[ ]:

# Euclidean Distance Matrix
N = responses.shape[0]
DM = np.zeros(rxx.shape)

for i in range(0,N):
    for j in range(i,N):
        DM[i,j] = np.sqrt(rxx[i,i]-2*rxx[i,j]+rxx[j,j])


# In[32]:

N


# In[ ]:

## dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))


# In[31]:

DM


# In[29]:

get_ipython().magic('who_ls')


# In[28]:

reset_selective rxx_diag,resp_spr


# In[ ]:

AA = pairwise_distances(resp_spr,metric='euclidean')


# In[ ]:




# In[ ]:




# In[ ]:


Rs = csr_matrix(responses.notnull().astype(int).values)


# In[ ]:

in_common = Rs.dot(Rs.T)


# In[ ]:

A = in_common[1,:].todense()
B = in_common[2,:].todense()


# In[ ]:

in_common[4,4]


# In[ ]:

in_common.data


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

R = responses.notnull().astype(int).values


# In[ ]:

np.dot(R[1,:],R[1,:])


# In[ ]:

in_common.todense()


# In[ ]:

in_common = spr.dot(Rs,Rs.T.tocsr())
in_common


# In[ ]:

reset_selective R

