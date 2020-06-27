
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler, scale
from sklearn.utils import resample


# In[2]:


df = pd.read_csv('./baseline_data.csv',header=0)


# In[3]:


df.head()


# In[4]:


Catdata = df.loc[0:,'PTETHCAT':'APOE4'].join(df['PTGENDER']).join(df['imputed_genotype'])
Numdata = df.loc[0:,:'Thickness..thickinthehead..2035'].join(df['AGE']).join(df['PTEDUCAT']).join(df['MMSE'])
labels = df['DX.bl']


# In[5]:


for i in Catdata.columns:
    the_value = str(Catdata[i].mode().values[0])
    Catdata[i].replace('NaN',the_value,inplace = True)
    dummy_data = pd.get_dummies(Catdata[i], prefix=i+"_", drop_first=True)
    Catdata = pd.concat([Catdata, dummy_data], axis=1)
    Catdata.drop(i, axis=1, inplace=True)
#for i in Numdata.columns:
    #the_value = str(Numdata[i].median().values[0])
    #Numdata[i].replace('NaN',the_value,inplace = True)


# In[6]:


le = LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
labels = labels.astype('float')


# In[7]:


data = Numdata.join(Catdata)
data['labels'] = labels
#data = resample(data,n_samples=628, random_state = 1)


# # split, normalize and bootstrap

# In[8]:


from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = ([],[],[],[])
X_train, X_test, y_train, y_test = train_test_split(data.loc[:,:'imputed_genotype__True'], data['labels'], test_size = 0.2, random_state = 2018)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train))
X_test = scaler.transform(X_test)
X_train['labels'] = np.array(y_train)


# In[9]:


from sklearn.utils import resample
Xs_train = []
Labels = []
for _ in range(5):
    d = resample(X_train,n_samples=502)
    Xs_train.append(d.iloc[:,:-1])
    Labels.append(d['labels'])


# # normalize and split

# In[10]:


from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = ([],[],[],[])
scaler = StandardScaler()
scaler.fit(data)
data = pd.DataFrame(scaler.transform(data))
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 1)
#DATA = X_train
#DATA['labels'] = y_train


# In[11]:


from sklearn.utils import resample
Xs_train = []
Labels = []
for _ in range(5):
    d = resample(X_train,n_samples=502)
    Xs_train.append(d.iloc[:,:-1])
    Labels.append(d['labels'])


# # Gaussian Process Regression

# A task that was considerably more difficult was the prediction of MMSE, which itself is a very noisy measure [8]. We used Gaussian Process Regression[5] (GPR) to predict this measure using selected imaging features, demographic features as well as genetic features. The choice of GPR was motivated by the ability of such models to fit very noisy data [4].

# In[21]:


from sklearn.gaussian_process import GaussianProcessClassifier
gp = GaussianProcessClassifier(optimizer='Welch',multi_class='one_vs_rest',n_restarts_optimizer=5)


# In[22]:


gpt = []
for i in range(5):
    gp.fit(Xs_train[i],Labels[i])
    gpt.append(gp.predict(X_test))


# In[23]:


gpc = []
for i in range(len(y_test)):
    p = []
    for j in gpt:
        p.append(j[i])
    counts = np.bincount(p)
    mode = np.argmax(counts)
    gpc.append(mode)


# In[24]:


count = 0
for i in range(len(gpc)):
    if gpc[i] == np.array(y_test)[i]:
        count += 1
accuracy = count/len(gpc)


# In[25]:


accuracy

