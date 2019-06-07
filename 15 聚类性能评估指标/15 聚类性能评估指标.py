import numpy as np

#############
# 外部指标度量
############

def calc_JC(y_true, y_pred):
    '''
    计算并返回JC系数
    :param y_true: 参考模型给出的簇，类型为ndarray
    :param y_pred: 聚类模型给出的簇，类型为ndarray
    :return: JC系数
    '''

    #******** Begin *******#
    a=0;b=0;c=0;d = 0
    for i in range(0, len(y_true)-1):
        for j in range(i+1,len(y_true)):
            if y_true[i]==y_true[j] and (y_pred[i]==y_pred[j]):
                a += 1
            elif y_true[i]!=y_true[j] and y_pred[i]==y_pred[j]:
                b += 1
            elif y_true[i]==y_true[j] and y_pred[i]!=y_pred[j]:
                c += 1
            else:
                d += 1
    return a / (a+b+c)
    #******** End *******#


def calc_FM(y_true, y_pred):
    '''
    计算并返回FM指数
    :param y_true: 参考模型给出的簇，类型为ndarray
    :param y_pred: 聚类模型给出的簇，类型为ndarray
    :return: FM指数
    '''

    #******** Begin *******#
    a=0;b=0;c=0;d = 0
    for i in range(0, len(y_true)-1):
        for j in range(i+1,len(y_true)):
            if y_true[i]==y_true[j] and (y_pred[i]==y_pred[j]):
                a += 1
            elif y_true[i]!=y_true[j] and y_pred[i]==y_pred[j]:
                b += 1
            elif y_true[i]==y_true[j] and y_pred[i]!=y_pred[j]:
                c += 1
            else:
                d += 1

    return (a**2 / ((a+b)*(a+c)))**0.5

    #******** End *******#

def calc_Rand(y_true, y_pred):
    '''
    计算并返回Rand指数
    :param y_true: 参考模型给出的簇，类型为ndarray
    :param y_pred: 聚类模型给出的簇，类型为ndarray
    :return: Rand指数
    '''

    #******** Begin *******#
    a=0;b=0;c=0;d = 0
    for i in range(0, len(y_true)-1):
        for j in range(i+1,len(y_true)):
            if y_true[i]==y_true[j] and (y_pred[i]==y_pred[j]):
                a += 1
            elif y_true[i]!=y_true[j] and y_pred[i]==y_pred[j]:
                b += 1
            elif y_true[i]==y_true[j] and y_pred[i]!=y_pred[j]:
                c += 1
            else:
                d += 1

    return 2*(a+d) / (len(y_true)*(len(y_true) - 1))
    #******** End *******#

t = [0,0,0,1,1,1]
p = [0,0,1,1,2,2]
# print(range(len(t)))
print(calc_JC(t,p),calc_FM(t,p),calc_Rand(t,p))



#############
# 内部指标度量
############

import numpy as np

def calc_DBI(feature, pred):
    '''
    计算并返回DB指数
    :param feature: 待聚类数据的特征，类型为`ndarray`
    :param pred: 聚类后数据所对应的簇，类型为`ndarray`
    :return: DB指数
    '''

    #********* Begin *********#


    #********* End *********#


def calc_DI(feature, pred):
    '''
    计算并返回Dunn指数
    :param feature: 待聚类数据的特征，类型为`ndarray`
    :param pred: 聚类后数据所对应的簇，类型为`ndarray`
    :return: Dunn指数
    '''

    #********* Begin *********#

    #********* End *********#

f = [[3,4],[6,9],[2,3],[3,4],[7,10],[8,11]]
p = [1, 2, 1, 1, 2, 2]









