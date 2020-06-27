
# coding: utf-8

# In[83]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler, scale


# In[84]:


df = pd.read_csv('./baseline_data.csv',header=0)


# In[85]:


df.head()


# In[86]:


Catdata = df.loc[0:,'PTETHCAT':'APOE4'].join(df['PTGENDER']).join(df['imputed_genotype'])
Numdata = df.loc[0:,:'Thickness..thickinthehead..2035'].join(df['AGE']).join(df['PTEDUCAT']).join(df['MMSE'])
labels = df['DX.bl']


# In[87]:


for i in Catdata.columns:
    the_value = str(Catdata[i].mode().values[0])
    Catdata[i].replace('NaN',the_value,inplace = True)
    if i != 'class':
        dummy_data = pd.get_dummies(Catdata[i], prefix=i+"_", drop_first=True)
        Catdata = pd.concat([Catdata, dummy_data], axis=1)
        Catdata.drop(i, axis=1, inplace=True)
    elif i == 'class':
        dummy_data = pd.get_dummies(Catdata[i])
        dummy_data.drop("notckd", axis=1, inplace=True)
        dummy_data.rename(columns={"ckd": "class"}, inplace=True)
        Catdata.drop(i, axis=1, inplace=True)
        Catdata = pd.concat([Catdata, dummy_data], axis=1)


# In[88]:


le = LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
labels = labels.astype('float')


# In[89]:


data = Numdata.join(Catdata)


# In[90]:


data.head()


# # split and normalize

# In[91]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 1)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train))
X_test = scaler.transform(X_test)


# # normalize and split

# In[92]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler, scale
scaler = StandardScaler()
scaler.fit(data)
data = pd.DataFrame(scaler.transform(data))
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 1)
X_train = pd.DataFrame(scaler.transform(X_train))

