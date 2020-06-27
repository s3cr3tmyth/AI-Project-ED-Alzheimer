
# coding: utf-8

# In[169]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler, scale


# In[170]:


df = pd.read_csv('./baseline_data.csv',header=0)


# In[171]:


df.head()


# In[172]:


Catdata = df.loc[0:,'PTETHCAT':'APOE4'].join(df['PTGENDER']).join(df['imputed_genotype'])
Numdata = df.loc[0:,:'Thickness..thickinthehead..2035'].join(df['AGE']).join(df['PTEDUCAT']).join(df['MMSE'])
labels = df['DX.bl']


# In[173]:


for i in Catdata.columns:
    the_value = str(Catdata[i].mode().values[0])
    Catdata[i].replace('NaN',the_value,inplace = True)
    dummy_data = pd.get_dummies(Catdata[i], prefix=i+"_", drop_first=True)
    Catdata = pd.concat([Catdata, dummy_data], axis=1)
    Catdata.drop(i, axis=1, inplace=True)
#for i in Numdata.columns:
    #the_value = str(Numdata[i].median().values[0])
    #Numdata[i].replace('NaN',the_value,inplace = True)


# In[174]:


le = LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
labels = labels.astype('float')


# In[216]:


data = Numdata.join(Catdata)


# # split, normalize and bootstrap

# In[220]:


from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = ([],[],[],[])
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 2018)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train))
X_test = scaler.transform(X_test)
X_train['labels'] = np.array(y_train)


# In[198]:


from sklearn.utils import resample
Xs_train = []
Labels = []
for _ in range(5):
    d = resample(X_train,n_samples=502)
    Xs_train.append(d.iloc[:,:-1])
    Labels.append(d['labels'])


# # normalize and split

# In[143]:


from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = ([],[],[],[])
scaler = StandardScaler()
scaler.fit(data)
data = pd.DataFrame(scaler.transform(data))
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 1)
#DATA = X_train
#DATA['labels'] = y_train


# In[84]:


from sklearn.utils import resample
Xs_train = []
Labels = []
for _ in range(5):
    d = resample(X_train,n_samples=502)
    Xs_train.append(d.iloc[:,:-1])
    Labels.append(d['labels'])
