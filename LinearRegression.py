#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
from numpy.linalg import matrix_rank
from numpy.linalg import inv

def lossFunc(x, w, y):
    loss = np.sum(np.power((np.dot(x, w) - y), 2)) / (2 * len(x))
    return loss

def gradientDescent(x,y):
    yita = 0.01  # 学习率
    add1 = np.ones((len(x),1))
    X = np.hstack((x, add1))
    w = np.zeros((len(X), 1))
    b = 0
    sita = np.vstack((w, b))
    for i in range(1000):
        if lossFunc(X, sita, y) < 0.00001:
            break
        else:
            sita = sita - yita * np.transpose(np.dot(np.transpose((np.dot(X, sita) - y)),X)) / len(X)
    return sita           
    

def linearRegression(x,y):
    # x 以一行作为数据，y为一列数据
    add1 = np.ones((len(x),1))
    X = np.hstack((x, add1))
    XTX = np.dot(np.transpose(X),X)
    if(matrix_rank(XTX) == len(XTX)):
        # 可逆则直接计算出最优值
        W = np.dot(np.dot(inv(XTX), np.transpose(X)), y)
        return W
    else:
        # 不可逆则用梯度下降法
        W = gradientDescent(x,y)
        return W


# In[ ]:




