#!/usr/bin/env python
# coding: utf-8

# In[193]:


# 可用于多分类的逻辑回归
import matplotlib.pyplot as plt
import numpy as np
def sigmoid(x, w):
    z = np.dot(x,w)
    y = 1/(1 + np.exp(-z))
    return y

def lossFunc(X, W, Y):
    loss = -1/len(X)*np.sum(Y * np.log(sigmoid(X, W))+ (1 - Y) * np.log(1 - sigmoid(X, W)))
    return loss
    
def GradientDescent(X, w, y):
    alpha = 0.01
    for i in range(1000):
        if lossFunc(X, w, y)<0.0001:
            break
        else:
            w = w - alpha * np.transpose(np.dot(np.transpose(sigmoid(X, w) - y),X))
    return w
    
def LogisticRegression(x, y):
    add1 = np.ones((len(x),1))
    X = np.hstack((x,add1))
    w = np.zeros((len(X[0]),1))
    W = GradientDescent(X, w, y)
    return W
    


# In[207]:


x_0 = np.zeros((np.sum(y==0),2))
x_1 = np.zeros((np.sum(y==1),2))
count1 = 0
count0 = 0
for i in range(len(y)):
    if y[i] == 1:
        x_1[count1,:] = x[i,:]
        count1 = count1 + 1
    else:
        x_0[count0,:] = x[i,:]
        count0 = count0 + 1
plt.scatter(x_1[:,0],x_1[:,1])
plt.scatter(x_0[:,0],x_0[:,1])


# In[210]:


data = np.loadtxt("C:\\Users\\97127\\Desktop\\ex2data2.txt", delimiter=",", dtype=np.float32)
x = data[:,0:2]
#  根据数据图像，添加自变量的平方项
x_append = np.power(x,2)
x = np.hstack((x,x_append))
y = data[:,2].reshape((len(data),1))
W = LogisticRegression(x, y)


# In[213]:


add1 = np.ones((len(x),1))
X = np.hstack((x,add1))
print(lossFunc(X, W, y))


# In[209]:


x_0 = np.zeros((np.sum(y==0),2))
x_1 = np.zeros((np.sum(y==1),2))
count1 = 0
count0 = 0
for i in range(len(y)):
    if y[i] == 1:
        x_1[count1,:] = x[i,:]
        count1 = count1 + 1
    else:
        x_0[count0,:] = x[i,:]
        count0 = count0 + 1
plt.scatter(x_1[:,0],x_1[:,1])
plt.scatter(x_0[:,0],x_0[:,1])


# In[217]:


data = np.loadtxt("C:\\Users\\97127\\Desktop\\ex2data1.txt", delimiter=",", dtype=np.float32)
x = data[:,0:2]
#x_append = np.power(x,-1)
#x = np.hstack((x,x_append))
x = np.power(x,-1)
y = data[:,2].reshape((len(data),1))
W = LogisticRegression(x, y)
print(W)


# In[218]:


add1 = np.ones((len(x),1))
X = np.hstack((x,add1))
print(lossFunc(X, W, y))


# In[ ]:




