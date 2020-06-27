
# coding: utf-8

# In[22]:


"""
A class to read the XML doc with each MRI scan. Each scan is included with an XML with various information.
"""

# parser
import xml.etree.ElementTree as ET
import os


class XMLReader:
    # Instantiates
    def __init__(self, path):
        # Safety check to make sure the file path is valid
        try:
            self.tree = ET.parse(path)
            self.root = self.tree.getroot()
        except IOError as ioerr:
            print("Failed to parse\n")
            print(ioerr)
            print("\n")

    # Returns 0 for NC, 1 for MCI, 2 for AD
    def subject_status(self):
        for i in self.root.iter("researchGroup"):
            if i.text == "CN":
                return 0
            elif i.text == "MCI":
                return 1
            elif i.text == "AD":
                return 2

    # Returns patient number
    def subject_identifier(self):
        for i in self.root.iter("subjectIdentifier"):
            return i.text

    # checks to see if the current XML doc is for an MRI
    def is_mri(self):
        for i in self.root.iter("modality"):
            if i.text == "MRI":
                return True

    # finds the image ID
    def getderiveduid(self):
        for i in self.root.iter("imageUID"):
            return i.text

    # finds a path to the respective scan from current directory
    def path_to_scan(self, origin):
        # first folder, patient number
        id = self.subject_identifier()
        path = "/" + id + "/"
        # second folder, scan label
        for i in self.root.iter("processedDataLabel"):
            label = i.text.split(";")
            break
        firstItem = True
        for i in label:
            if firstItem == True:
                path = path + i.replace(" ", "")
                firstItem = False
                continue
            path = path + "__" + i.strip().replace(" ", "_")
        
        '''
        # third folder scan date
        for i in self.root.iter("dateAcquired"):
            split = i.text.split(" ")
            lhs = split[0]
            rhs = split[1]
            break
        path = path + "/" + lhs
        rhsplit = rhs.split(":")
        for i in rhsplit:
            path = path + "_" + i
        '''
        # third folder scan date
        item3 = os.listdir(origin + path)
        loc = []
        for i in item3:
            if 'DS' not in i:
                loc.append(i)
        path = path + "/" + loc[0]
        
        # fourth folder, series number
        for i in self.root.iter("seriesIdentifier"):
            sid = i.text
            break
        path = path + "/" + "S" + sid
        
        
        # finally finds the scan, checks to see if its a .nii file
        items = os.listdir(origin + path)
        scans = 0
        scan = []
        for i in items:
            curr = i.split(".")
            if curr[len(curr) - 1] == "nii":
                scan.append(i)
                scans += 1
        if scans > 1:
            imageid = self.getderiveduid()
            for i in scan:
                parsed = i.replace(".nii", "").split("_")
                for x in parsed:
                    if x == imageid:
                        return path + "/" + i
        return origin + path + "/" + scan[0]


# In[23]:


def inputs(eval_data):
    if eval_data is True:
        data_dir = "./ADNI/test_data"
    else:
        data_dir = "./ADNI/train_data"
    return img_inputs(eval_data=eval_data, data_dir=data_dir,batch_size=3)
datadiction = {}


# In[24]:


import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import nibabel as nib
from tensorflow.python.platform import gfile

width = 40
height = 40
depth = 20
batch_index = 0
ims = []
filenames = []
# user selection
num_class = 3


def img_inputs(eval_data, data_dir):
    global ims
    if not eval_data:
        all = os.listdir(data_dir)
        xmls = []
        for x in all:
            temp = x.split(".")
            if temp[len(temp)-1] == "xml":
                xmls.append(x)
        for f in xmls:
            r = XMLReader(data_dir+'/'+f)
            p = r.path_to_scan(data_dir)
            la = r.subject_status()
            if not gfile.Exists(p):
                raise ValueError('Failed to find file: ' + f)
            else:
                ims.append(str(la)+'||'+ p)
    else:
        all = os.listdir(data_dir)
        xmls = []
        for x in all:
            temp = x.split(".")
            if temp[len(temp)-1] == "xml":
                xmls.append(x)
        for f in xmls:
            r = XMLReader(data_dir+'/'+f)
            p = r.path_to_scan(data_dir)
            la = r.subject_status()
            if not gfile.Exists(p):
                raise ValueError('Failed to find file: ' + f)
            else:
                ims.append(str(la)+'||'+p)
    random.shuffle(ims)
    return ims

def inputs(eval_data):
    if eval_data is True:
        data_dir = "./ADNI/test_data"
    else:
        data_dir = "./ADNI/train_data"
    return img_inputs(eval_data=eval_data, data_dir=data_dir)


def get_image(sess, eval_data, batch_size):
    global batch_index, filenames
    if eval_data == True:
        filenames = []
    if len(filenames) == 0: 
        filenames = inputs(eval_data) 
    Max = len(filenames)
    begin = batch_index
    end = batch_index + batch_size
    avail = batch_size
    if end >= Max:
        end = Max
        batch_index = 0
    x_data = np.array([], np.float32)
    y_data = np.zeros((batch_size, num_class)) # zero-filled list for 'one hot encoding'
    index = 0
    for i in range(begin, end):
        imagePath = filenames[i].split('||')[1]
        try: 
            image = datadiction[imagePath]
        except:
            FA_org = nib.load(imagePath)
            FA_data = FA_org.get_data()[:,:,90:110].astype('float32')  # 256x256x166; numpy.ndarray
            # TensorShape([Dimension(256), Dimension(256), Dimension(166)])                       
            resized_image = tf.image.resize_images(images=FA_data, size=(width,height), method=1)
            image = sess.run(resized_image)  # (256,256,166)
            datadiction[imagePath] = image
        x_data = np.append(x_data, np.asarray(image, dtype='float32')) # (image.data, dtype='float32')
        y_data[index][int(filenames[i].split('||')[0])] = 1.0  # assign 1 to corresponding column (one hot encoding)
        index+=1
    batch_index += batch_size  # update index for the next batch
    x_data_ = x_data.reshape(avail, height * width * depth)
    return  x_data_, y_data


# In[36]:


nLabel = 3

# Start TensorFlow InteractiveSession
import tensorflow as tf
sess = tf.InteractiveSession()

# Placeholders (MNIST image:28x28pixels=784, label=10)
xx = tf.placeholder(tf.float32, shape=[None, width*height*depth]) # [None, 28*28]
yy = tf.placeholder(tf.float32, shape=[None, nLabel])  # [None, 10]

def Weight(shape):
    dist = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(dist)

def Bias(shape):
    dist = tf.constant(0.1,shape=shape)
    return tf.Variable(dist)

def Convolution(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME') # conv2d, [1, 1, 1, 1]


def Max_pool(x): 
    return tf.nn.max_pool3d(x, ksize=[1, 4, 4, 4, 1], strides=[1, 4, 4, 4, 1], padding='SAME')


W_conv1 = Weight([5, 5, 5, 1, 32])  
b_conv1 = Bias([32]) 


x_image = tf.reshape(xx, [-1,width,height,depth,1]) 
print(x_image.get_shape) 

h_conv1 = tf.nn.relu(Convolution(x_image, W_conv1) + b_conv1)  
print(h_conv1.get_shape) 
h_pool1 = Max_pool(h_conv1)  
print(h_pool1.get_shape) 


W_conv2 = Weight([5, 5, 5, 32, 64]) 
b_conv2 = Bias([64]) # [64]

h_conv2 = tf.nn.relu(Convolution(h_pool1, W_conv2) + b_conv2)  
print(h_conv2.get_shape) 
h_pool2 = Max_pool(h_conv2) 
print(h_pool2.get_shape) 


W_fc1 = Weight([3*3*2*64, 1024]) 
b_fc1 = Bias([1024]) # [1024]]

h_pool2_flat = tf.reshape(h_pool2, [-1, 3*3*2*64]) 
print(h_pool2_flat.get_shape)  
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  
print(h_fc1.get_shape)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
print(h_fc1_drop.get_shape)  # -> output: 1024

W_fc2 = Weight([1024, nLabel]) # [1024, 10]
b_fc2 = Bias([nLabel]) # [10]

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print(y_conv.get_shape)  


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=yy, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)  # 1e-4
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(yy,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[37]:


sess.run(tf.global_variables_initializer())

# Include keep_prob in feed_dict to control dropout rate.
for i in range(100):
    batch = get_image(sess,False,3)
    feature = batch[0]
    ylabel = batch[1]
    print(i)
    # Logging every 100th iteration in the training process.
    if i%3 == 0:
        train_accuracy = accuracy.eval(feed_dict={xx:feature, yy:ylabel, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        print('cost:',sess.run(loss,feed_dict={xx:feature,yy:ylabel,keep_prob:1.0}))
        testset = get_image(sess,True,9)
        print("test accuracy %g"%accuracy.eval(feed_dict={xx: testset[0], yy: testset[1], keep_prob: 1.0}))
    train_step.run(feed_dict={xx: feature, yy: ylabel, keep_prob: 0.5})

# Evaulate our accuracy on the test data
testset = get_image(sess,True,9)
print("test accuracy %g"%accuracy.eval(feed_dict={xx: testset[0], yy: teseset[1], keep_prob: 1.0}))

