import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def scale_feature(x, feature_range=(-1, 1)):
    x = np.asarray(x, dtype=np.float)
    if x.ndim < 2:
        x = x[np.newaxis, :]
    assert x.ndim == 2, '数据的维度超过了2'
    x_min = [np.min(x[:, i]) for i in range(x.shape[1])]
    x_max = [np.max(x[:, j]) for j in range(x.shape[1])]
    x_scaled = np.zeros(x.shape)
    for i in range(x.shape[1]):
        x[:, i] = (x[:, i] - x_min[i])/(x_max[i] - x_min[i])
        x_scaled = x * (feature_range[1] - feature_range[0]) + feature_range[0]

    return x_scaled

def split(x, y, ratio, random_seed=0):
    np.random.seed(random_seed)
    indices = np.random.permutation(len(x))
    train_index = indices[:-int(len(x)*ratio)]
    test_index = indices[-int(len(x)*ratio):]

    x_train, y_train = x[train_index, :], y[train_index]
    x_test, y_test = x[test_index, :], y[test_index]

    return x_train, x_test, y_train, y_test

def visualize(x, y):
    pos = np.where(y)[0]      # 查找标签不为0的样本的索引
    neg = np.where(y==0)[0]
    plt.scatter(x[pos, 0], x[pos, 1], marker='o', c='b')
    plt.scatter(x[neg, 0], x[neg, 1], marker='x', c='r')
    plt.xlabel('grade1')
    plt.ylabel('grade2')

    plt.legend(['positive instance', 'negative instance'])
    plt.show()

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def predict(theta, x):
    z = np.sum(np.multiply(x, theta), axis=1)
    return sigmoid(z)

def cross_entropy(y, y_pred):
    loss = -y*np.log(y_pred)-(1-y)*np.log(1-y_pred)
    return loss.sum() / len(y)

def derivative(x, y_pred, y):
    n = len(x)
    d_theta = np.mean([x[i]*(y_pred[i] - y[i]) for i in range(n)],axis=0)

    return d_theta

def gradient_descent(x, y_pred, y, theta, lr):
    theta -= lr * derivative(x, y_pred, y)

    return theta

def init_theta(dims):
    theta = np.random.uniform(0, 1, dims)

    return theta

def Logistic_regression(x, y, theta, lr):
    y_pred = predict(theta, x)
    theta = gradient_descent(x, y_pred, y, theta, lr)

    return theta

def main():
    df = pd.read_csv('data.csv', header=0)
    x = df[[df.columns[0], df.columns[1]]]
    x = scale_feature(x)
    y = np.asarray(df[df.columns[2]].map(lambda x: x.strip(';')), dtype='float')

    # To visualize datasets, uncomment the following line
    visualize(x, y)
    x_train, x_test, y_train, y_test = split(x, y, 0.2)

    learning_rate = 0.01
    num_epochs = 500
    theta = init_theta(x_train.shape[-1])

    for epoch in range(num_epochs):
        y_train_pred = predict(theta, x_train)
        y_test_pred = predict(theta, x_test)
        theta = Logistic_regression(x_train, y_train, theta, lr=learning_rate)
        losses = cross_entropy(y_train, y_train_pred)
        acc_train = np.mean(np.round(y_train_pred) == y_train)
        acc_test = np.mean(np.round(y_test_pred) == y_test)

        if epoch % 20 == 19:
            print('训练集的loss:{:.4f}, accuracy:{} | 测试集的准确率: {}'.format(losses, acc_train, acc_test))

main()