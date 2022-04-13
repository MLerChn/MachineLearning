import math

# 机器学习 ≈ 找函数
# 专业名词：两大类任务
#  回归regression：结果是一个scalar（数值）
#  分类classification：从设定好的选项里选择一个作为结果输出

# How to find a function?
#  Step1: 定义带未知参数的函数
#         y = b + w * x1     y:output  b:bias  w:weight  x1:input

#  Step2: 定义损失函数，用于评估结果的好坏
#         L(b,w) = 1 / N * ∑ e[n]
#             MAE: e[i] = abs(y[i] - y_hat[i])
#             MSE: e[i] = (y[i] - y_hat[i])**2

#  Step3: 找w*,b*使L(b,w)最小——方法：gradient descent梯度下降法
#         SStep1：视为只有w这个参数
#         SStep2：随机取值w = w0
#         SStep3：L对w求导，然后代入w = w0，b = b0 结果为partial_Lw；
#                 L对b求导，然后代入w = w0，b = b0 结果为partial_Lb
#         SStep4: w1 = w0 - β * partial_Lw
#                 b1 = b0 - β * partial_Lb
#         SStep5: 判断结束条件

# 激活函数类型：激活函数 = hidden layer
# 1、sigmoid函数
#            y = b + ∑c[i]*sigmoid(b[i] + ∑w[i][j]*x[j])
# 改变constant：曲线上下平移
# 改变bias：曲线左右平移
# 改变weight：曲线中间部分斜率改变
def sigmoid(x):
    y = 1 / (1 + math.exp(-x))
    return y

#  2、ReLU函数：两个ReLU≈一个sigmoid
def ReLU(x, alpha):
    if x > 0:
        y = alpha * x
    else:
        y = 0
    return y

#  deep learning = many hidden layer

