
# coding: utf-8

# In[1]:


import numpy as np
import random
import tensorflow as tf
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler, scale
df = pd.read_csv('./baseline_data.csv',header=0)


# In[2]:


Catdata = df.loc[0:,'PTETHCAT':'APOE4'].join(df['PTGENDER']).join(df['imputed_genotype'])
Numdata = df.loc[0:,:'Thickness..thickinthehead..2035'].join(df['AGE']).join(df['PTEDUCAT']).join(df['MMSE'])
labels = df['DX.bl']


# In[3]:


for i in Catdata.columns:
    the_value = str(Catdata[i].mode().values[0])
    Catdata[i].replace('NaN',the_value,inplace = True)
    dummy_data = pd.get_dummies(Catdata[i], prefix=i+"_", drop_first=True)
    Catdata = pd.concat([Catdata, dummy_data], axis=1)
    Catdata.drop(i, axis=1, inplace=True)
#for i in Numdata.columns:
    #the_value = str(Numdata[i].median().values[0])
    #Numdata[i].replace('NaN',the_value,inplace = True)


# In[4]:


le = LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
labels = labels.astype('float')


# In[5]:


from sklearn.utils import resample
data = Numdata.join(Catdata)
data['labels'] = labels
#data = resample(data,n_samples=628, random_state = 2018)
data.head()


# In[6]:


from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = ([],[],[],[])
X_train, X_test, y_train, y_test = train_test_split(data.loc[:,:'imputed_genotype__True'], data['labels'], test_size = 0.2, random_state = 2018)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train))
X_test = scaler.transform(X_test)


# In[7]:


X_train.head()


# In[8]:


trainX = np.array(X_train)
trainy = np.array(y_train).reshape(-1,1)

testX = np.array(X_test)
testy = np.array(y_test).reshape(-1,1)


# In[20]:


x = tf.placeholder(tf.float32,[None,2161])
y = tf.placeholder(tf.float32,[None,1])

W1 = tf.Variable(tf.truncated_normal([2161,200],stddev=0.1))
b1 = tf.Variable(tf.zeros([200])+0.1)
z1 = tf.matmul(x,W1)+b1
h1 = tf.nn.relu(z1)

W2 = tf.Variable(tf.truncated_normal([200,50],stddev=0.1))
b2 = tf.Variable(tf.zeros([50])+0.1)
z2 = tf.matmul(z1,W2)+b2
h2 = tf.nn.tanh(z2)

W3 = tf.Variable(tf.truncated_normal([50,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10])+0.1)
z3 = tf.matmul(h2,W3)+b3
h3 = tf.nn.tanh(z3)

W4 = tf.Variable(tf.truncated_normal([10,3],stddev=0.1))
b4 = tf.Variable(tf.zeros([3])+0.1)
z4 = tf.matmul(h3,W4)+b4

y_prediction = tf.nn.softmax(z4)

#loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_prediction), reduction_indices=[1]))
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = y_prediction))
loss = tf.reduce_mean((y-y_prediction)**2)

#Train = tf.train.GradientDescentOptimizer(0.0005).minimize(loss) #DOM:0.00001
Train = tf.train.MomentumOptimizer(learning_rate=0.5,momentum = 0.9).minimize(loss)
#Train = tf.train.AdamOptimizer(1e-3).minimize(loss)

init = tf.global_variables_initializer()

correct_p = tf.equal(tf.argmax(y,1),tf.argmax(y_prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_p,tf.float32))

# using validation set to determine hyperparameters.
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1500):
        sess.run(Train,feed_dict={x:trainX,y:trainy})
        acc = sess.run(accuracy,feed_dict={x:trainX, y:trainy})
        tacc = sess.run(accuracy,feed_dict={x:testX, y:testy})
        cost = sess.run(loss,feed_dict={x:trainX,y:trainy})
        #print('batch: '+ str(batch)+',node: '+str(node)+', learning rate: '+str(lr))
        print('Iter'+str(epoch+1)+',Training loss: '+str(cost)+', Accuracy: '+str(acc)+',TAccuracy:'+str(tacc))

