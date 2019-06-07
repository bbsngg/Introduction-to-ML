#!/usr/bin/env python
# coding: utf-8

# # HW4 logistic_regression（逻辑回归作业练习）
# 
# ## <font color=brown> 本次作业包括两个练习</font>
# ## 练习1: 使用逻辑回归模型来对学生是否能被大学录取进行预测。
# ## 已知数据集中包含两次测试成绩exam1 和 exam2，以及是否被录取的标记 admitted。

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report #这个包是评价报告
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')


# # 1.1 准备数据

# In[3]:


print("171250628 宋定杰")  #  双引号内替换成你的学号和姓名
data = pd.read_csv('hw4data1.txt', names=['exam1', 'exam2', 'admitted'])
data.head()#看前五行


# In[4]:


data.describe()


# In[7]:


# 关于 SeaBorn 画图，可参考：https://www.jianshu.com/p/5ff47c7d0cc9

sns.set(context="notebook", style="darkgrid", palette=sns.color_palette("RdBu", 2),color_codes=False)

sns.lmplot('exam1', 'exam2', hue='admitted', data=data, 
           size=6, 
           fit_reg=False, 
           scatter_kws={"s": 50}
          )
plt.show()#看下数据的样子


# In[8]:


def get_X(df):#读取特征
#     """
#     use concat to add intersect feature to avoid side effect
#     not efficient for big dataset though
#     """
    ones = pd.DataFrame({'ones': np.ones(len(df))}) # ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)            # 合并数据，根据列合并
    return data.iloc[:, :-1].as_matrix()            # 这个操作返回 ndarray,不是矩阵


def get_y(df):#读取标签
#     '''assume the last column is the target'''
    return np.array(df.iloc[:, -1])                # df.iloc[:, -1]是指df的最后一列


def normalize_feature(df):
#     """Applies function along input axis(default 0) of DataFrame."""
    return df.apply(lambda column: (column - column.mean()) / column.std())  # 特征缩放


# In[9]:


X = get_X(data)
print(X.shape)

y = get_y(data)
print(y.shape)


# # 1.2 sigmoid 函数
# g 代表一个常用的逻辑函数（logistic function）为S形函数（Sigmoid function），公式为： \\[g\left( z \right)=\frac{1}{1+{{e}^{-z}}}\\] 
# 合起来，我们得到逻辑回归模型的假设函数： 
# 	\\[{{h}_{\theta }}\left( x \right)=\frac{1}{1+{{e}^{-{{\theta }^{T}}X}}}\\] 
# 

# In[13]:


## 实现sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))                               ## 补充YOUR_CODE处的代码


# In[14]:


fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(-10, 10, step=0.01),
        sigmoid(np.arange(-10, 10, step=0.01)))
ax.set_ylim((-0.1,1.1))
ax.set_xlabel('z', fontsize=18)
ax.set_ylabel('g(z)', fontsize=18)
ax.set_title('sigmoid function', fontsize=18)
plt.show()


# # 1.3 cost function(代价函数)
# > * $max(\ell(\theta)) = min(-\ell(\theta))$  
# > * choose $-\ell(\theta)$ as the cost function
# 
# $$\begin{align}
#   & J\left( \theta  \right)=-\frac{1}{m}\sum\limits_{i=1}^{m}{[{{y}^{(i)}}\log \left( {{h}_{\theta }}\left( {{x}^{(i)}} \right) \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-{{h}_{\theta }}\left( {{x}^{(i)}} \right) \right)]} \\ 
#  & =\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( {{h}_{\theta }}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1-{{h}_{\theta }}\left( {{x}^{(i)}} \right) \right)]} \\ 
# \end{align}$$
# 

# In[15]:


theta = theta=np.zeros(3) # X(m*n) so theta is n*1
theta


# In[16]:


## 实现代价函数 cost
def cost(theta, X, y):
    thetaX = np.dot(X,theta.T)
    in1 = np.dot(-y, np.log(sigmoid(thetaX)))
    in2 = np.dot(1-y, np.log(1 - sigmoid(thetaX)))
    inner = in1 - in2
    return np.sum(inner) / len(X)                               ## 补充YOUR_CODE处的代码

# X @ theta与X.dot(theta)等价


# In[17]:


cost(theta, X, y)


# # 1.4 gradient 函数(梯度函数)
# * 梯度计算$\frac{1}{m} X^T( Sigmoid(X\theta) - y )$
# $$\frac{\partial J\left( \theta  \right)}{\partial {{\theta }_{j}}}=\frac{1}{m}\sum\limits_{i=1}^{m}{({{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}})x_{_{j}}^{(i)}}$$

# In[ ]:


## 实现梯度计算函数 gradient
def gradient(theta, X, y):
    inner = np.dot(X.T, sigmoid(np.dot(X, theta.T))-y)
    return inner / len(X)                                  ## 补充YOUR_CODE处的代码


# In[ ]:


gradient(theta, X, y)


# # 1.5 拟合参数
# * <font color=Brow>这里我们不使用梯度下降法，改用 [`scipy.optimize.minimize`](http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize) 去计算参数</font>  
# 

# In[ ]:


import scipy.optimize as opt


# In[ ]:


res = opt.minimize(fun=cost, x0=theta, args=(X, y), method='Newton-CG', jac=gradient)


# In[ ]:


print(res)


# # 1.6 用训练集预测和验证

# In[ ]:


## 回忆课堂内容，此处的概率如何计算
def predict(x, theta):
    prob = sigmoid(np.dot(x,theta.T)) #YOUR_CODE#                               ## 补充YOUR_CODE处的代码
    return (prob >= 0.5).astype(int)


# In[ ]:


final_theta = res.x
y_pred = predict(X, final_theta)

print(classification_report(y, y_pred))
## 此处的classification_report函数将根据真值y与预测值y_pred，输出分析报告


# # 1.7 寻找决策边界
# 
# * $\theta^T x = 0$ (一个样本$x$)
# * $X \theta = 0$ (所有训练样本$X$)

# In[ ]:


print(res.x) # this is final theta
res.x[2]


# In[ ]:


#YOUR_CODE begin
    #此处补充4~5行代码，计算x和y的值，用于绘制决策边界
#YOUR_CODE end


# In[ ]:


data.describe()  # find the range of x and y


# In[ ]:


sns.set(context="notebook", style="ticks", font_scale=1.5)

sns.lmplot('exam1', 'exam2', hue='admitted', data=data, 
           size=6, 
           fit_reg=False, 
           scatter_kws={"s": 25}
          )

plt.plot(x, y, 'grey')
plt.xlim(0, 130)
plt.ylim(0, 130)
plt.title('Decision Boundary')
plt.show()


# # 2 正则化逻辑回归
# ## 练习2: 使用正则化逻辑回归模型来对芯片是否能通过质检进行预测。
# ## 已知数据集中包含两次检测成绩test1 和 test2，以及是否通过质检的标记 accepted。
# # 2.1 准备数据

# In[ ]:


df = pd.read_csv('hw4data2.txt', names=['test1', 'test2', 'accepted'])
df.head()


# In[ ]:


sns.set(context="notebook", style="ticks", font_scale=1.5)

sns.lmplot('test1', 'test2', hue='accepted', data=df, 
           size=5, 
           fit_reg=False, 
           scatter_kws={"s": 50}
          )

plt.title('Regularized Logistic Regression')
plt.show()


# # 2.2 feature mapping（特征映射）
# 
# * 我们进行多项式扩展来构造特征
# polynomial expansion
# 
# ```
# for i in 0..i
#   for p in 0..i:
#     output x^(i-p) * y^p
# ```
# <img style="float: left;" src="mapped_feature.png">

# In[ ]:


def feature_mapping(x, y, power, as_ndarray=False):
    #     """return mapped features as ndarray or dataframe"""
    # data = {}
    # # inclusive
    # for i in np.arange(power + 1):
    #     for p in np.arange(i + 1):
    #         data["f{}{}".format(i - p, p)] = np.power(x, i - p) * np.power(y, p)

    data = {"f{}{}".format(i - p, p): np.power(x, i - p) * np.power(y, p)
                for i in np.arange(power + 1)
                for p in np.arange(i + 1)
            }

    if as_ndarray:
        return pd.DataFrame(data).as_matrix()
    else:
        return pd.DataFrame(data)


# In[ ]:


x1 = np.array(df.test1)
x2 = np.array(df.test2)


# In[ ]:


data = feature_mapping(x1, x2, power=6)
print(data.shape)
data.head()


# In[ ]:


data.describe()


# # 2.3 regularized cost（正则化代价函数）
# $$J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( {{h}_{\theta }}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1-{{h}_{\theta }}\left( {{x}^{(i)}} \right) \right)]}+\frac{\lambda }{2m}\sum\limits_{j=1}^{n}{\theta _{j}^{2}}$$

# In[ ]:


theta = np.zeros(data.shape[1])
X = feature_mapping(x1, x2, power=6, as_ndarray=True)
print(X.shape)

y = get_y(df)
print(y.shape)


# In[ ]:


## 实现正则化代价函数
# def regularized_cost(theta, X, y, lambd=1):
# #YOUR_CODE begin
#     #此处补充3～4行代码，用于实现正则化代价函数
# #YOUR_CODE end
# #正则化代价函数
#
#
# # In[ ]:
#
#
# regularized_cost(theta, X, y, lambd=1)
#
#
# # # 2.4 regularized gradient(正则化梯度)
# # $$\frac{\partial J\left( \theta  \right)}{\partial {{\theta }_{j}}}=\left( \frac{1}{m}\sum\limits_{i=1}^{m}{\left( {{h}_{\theta }}\left( {{x}^{\left( i \right)}} \right)-{{y}^{\left( i \right)}} \right)} \right)+\frac{\lambda }{m}{{\theta }_{j}}\text{ }\text{             for  j}\ge \text{1}$$
#
# # In[ ]:
#
#
# ## 实现正则化梯度函数
# def regularized_gradient(theta, X, y, lambd=1):
# #YOUR_CODE begin
#     #此处补充4~5行代码，用于实现正则化梯度函数
# #YOUR_CODE end
#
#
# # In[ ]:
#
#
# regularized_gradient(theta, X, y)
#
#
# # # 2.5 拟合参数
#
# # In[ ]:
#
#
# import scipy.optimize as opt
#
#
# # In[ ]:
#
#
# print('init cost = {}'.format(regularized_cost(theta, X, y)))
#
# # 此处我们直接使用scipy中的optimize库来进行参数计算
# res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y), method='Newton-CG', jac=regularized_gradient)
# res
#
#
# # # 2.6 预测
#
# # In[ ]:
#
#
# final_theta = res.x
# y_pred = predict(X, final_theta)
#
# print(classification_report(y, y_pred))
#
#
# # # 2.7 使用不同的 $\lambda$ 对于预测结果的影响
# # * 尝试不同的lambda(0.1,10)对于上述2.6节预测结果的影响，将结果反映在作业报告中
# #
# # # 2.8 画出决策边界
# # * 我们找到所有满足 $X \theta = 0$ 的$x$
# # * 以下内容不需要完善代码，仅供同学们参考
#
# # In[ ]:
#
#
# def draw_boundary(power, lambd):
# #     """
# #     power: polynomial power for mapped feature
# #     lambd: lambda constant
# #     """
#     density = 1000
#     threshhold = 2 * 10**-3
#
#     final_theta = feature_mapped_logistic_regression(power, lambd)
#     x, y = find_decision_boundary(density, power, final_theta, threshhold)
#
#     df = pd.read_csv('hw4data2.txt', names=['test1', 'test2', 'accepted'])
#     sns.lmplot('test1', 'test2', hue='accepted', data=df, size=5, fit_reg=False, scatter_kws={"s": 100})
#
#     plt.scatter(x, y, c='R', s=10)
#     plt.title('Decision boundary')
#     plt.show()
#
#
# # In[ ]:
#
#
# def feature_mapped_logistic_regression(power, lambd):
# #     """for drawing purpose only.. not a well generealize logistic regression
# #     power: int
# #         raise x1, x2 to polynomial power
# #     lambd: int
# #         lambda constant for regularization term
# #     """
#     df = pd.read_csv('hw4data2.txt', names=['test1', 'test2', 'accepted'])
#     x1 = np.array(df.test1)
#     x2 = np.array(df.test2)
#     y = get_y(df)
#
#     X = feature_mapping(x1, x2, power, as_ndarray=True)
#     theta = np.zeros(X.shape[1])
#
#     res = opt.minimize(fun=regularized_cost,
#                        x0=theta,
#                        args=(X, y, lambd),
#                        method='TNC',
#                        jac=regularized_gradient)
#     final_theta = res.x
#
#     return final_theta
#
#
# # In[ ]:
#
#
# def find_decision_boundary(density, power, theta, threshhold):
#     t1 = np.linspace(-1, 1.5, density)
#     t2 = np.linspace(-1, 1.5, density)
#
#     cordinates = [(x, y) for x in t1 for y in t2]
#     x_cord, y_cord = zip(*cordinates)
#     mapped_cord = feature_mapping(x_cord, y_cord, power)  # this is a dataframe
#
#     inner_product = mapped_cord.as_matrix() @ theta
#
#     decision = mapped_cord[np.abs(inner_product) < threshhold]
#
#     return decision.f10, decision.f01
# #寻找决策边界函数
#
#
# # In[ ]:
#
#
# draw_boundary(power=6, lambd=1)#lambda=1
#
#
# # In[ ]:
#
#
# draw_boundary(power=6, lambd=0)  # no regularization, over fitting，#lambda=0,没有正则化，过拟合了
#
#
# # In[ ]:
#
#
# draw_boundary(power=6, lambd=100)  # underfitting，#lambda=100,欠拟合
#
#
# # In[ ]:
#
#
#
#
