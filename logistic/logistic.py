import numpy as np
import matplotlib.pyplot as plt

def load_data():
    '''
    :return: data_arr,label_arr
    '''
    data_arr = []
    label_arr = []
    file = open('../data/5.Logistic/TestSet.txt', 'r')
    for line in file:
        temp = line.strip().split()
        data_arr.append([1.0,np.float(temp[0]),np.float(temp[1])])
        label_arr.append(int(temp[2]))
    return data_arr,label_arr

def sigmoid(x):
    '''
    这里面如果x太大的话exp（x）会造成溢出的问题，使用一个bigfloat这个库会解决这个问题
    :param x:
    :return: sigmoid x
    '''
    return 1.0/(1 + np.exp(-x))


def grad_ascent(data_arr,label_arr):
    '''
    梯度上升优化算法，返回的是更改后的weights，其实这里面使用了极大似然法
    :param data_arr:
    :param label_arr:
    :return:weights
    '''
    data_mat = np.mat(data_arr)
    label_mat = np.mat(label_arr).transpose()
    m,n = np.shape(data_mat)
    weights = np.ones((n,1))
    alpha = 0.001
    #最大迭代次数
    max_cycles = 500
    for i in range(max_cycles):
        #这里是点乘 [m,n]*[n,1]
        h = sigmoid(data_mat * weights)
        error = label_mat - h
        #看一下推导
        weights = weights + alpha * data_mat.transpose() * error
    return weights


def grad_ascent1(data_arr, class_labels):
    """
    梯度上升法，其实就是因为使用了极大似然估计，这个大家有必要去看推导，只看代码感觉不太够
    :param data_arr: 传入的就是一个普通的数组，当然你传入一个二维的ndarray也行
    :param class_labels: class_labels 是类别标签，它是一个 1*100 的行向量。
                    为了便于矩阵计算，需要将该行向量转换为列向量，做法是将原向量转置，再将它赋值给label_mat
    :return:
    """
    # 注意一下，我把原来 data_mat_in 改成data_arr,因为传进来的是一个数组，用这个比较不容易搞混
    # turn the data_arr to numpy matrix
    data_mat = np.mat(data_arr)
    # 变成矩阵之后进行转置
    label_mat = np.mat(class_labels).transpose()
    # m->数据量，样本数 n->特征数
    m, n = np.shape(data_mat)
    # 学习率，learning rate
    alpha = 0.001
    # 最大迭代次数，假装迭代这么多次就能收敛2333
    max_cycles = 500
    # 生成一个长度和特征数相同的矩阵，此处n为3 -> [[1],[1],[1]]
    # weights 代表回归系数， 此处的 ones((n,1)) 创建一个长度和特征数相同的矩阵，其中的数全部都是 1
    weights = np.ones((n, 1))
    for k in range(max_cycles):
        # 这里是点乘  m x 3 dot 3 x 1
        h = sigmoid(data_mat * weights)
        error = label_mat - h
        # 这里比较建议看一下推导，为什么这么做可以，这里已经是求导之后的
        weights = weights + alpha * data_mat.transpose() * error
    return weights


def plot_best_fit(weights):
    '''
    可视化
    :param weights:
    :return:
    '''
    data_arr,label_arr = load_data()
    data_mat = np.array(data_arr)
    n = np.shape(data_mat)[1]
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for i in range(n):
        if label_arr[1] == 1:
            x1.append(data_mat[i,1])
            y1.append(data_mat[i,2])
        else:
            x2.append(data_arr[i,1])
            y2.append(data_arr[i,2])

    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.scatter(x1, y1, s=30, color='k', marker='^')
    ax.scatter(x2, y2, s=30, color='red', marker='s')

    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    print(np.shape(x),np.shape(weights[0]))

    ax.plot(x,y)
    plt.xlabel('xlabel')
    plt.ylabel('ylabel')
    plt.show()




if __name__ == '__main__':
    data_arr,label_arr = load_data()
    weights = grad_ascent1(data_arr,label_arr)
    plot_best_fit(weights)