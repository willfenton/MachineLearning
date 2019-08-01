#!/usr/bin/env python
#------------------------------------------------------------------------------------
# Author: Will Fenton
# Date: July 24 2019
# 
# Regularized Logistic Regression 
# Used for Binary Classification
# Regularization helps to avoid overfitting the data by penalizing large theta values
#------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#------------------------------------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost_function(theta, X, y, reg_param):
    m, n = X.shape
    eps = 1e-8

    h = sigmoid(X @ theta.T)
    reg_term = sum(theta.reshape(theta.size, 1)[1:, :] ** 2) * (reg_param / (2 * m))
    cost = ((-y.T  @ np.log(h + eps) - (1 - y).T @ np.log(1 - h + eps)) / m)[0] + reg_term

    return cost


def cost_gradient(theta, X, y, reg_param):
    m, n = X.shape

    h = np.vstack(sigmoid(X @ theta.T))
    reg_term = np.vstack((np.zeros((1, 1)), theta.reshape(theta.size, 1)[1:,:])) * (reg_param / m)
    gradient = ((X.T @ (h - y)) / m) + reg_term

    return gradient.T[0]


def map_feature(X1, X2):
    degree = 8
    out = np.ones((X1.size, 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            new_column = (X1 ** (i-j)) * (X2 ** j)
            out = np.hstack((out, new_column.reshape(X1.size, 1)))
    return out

#------------------------------------------------------------------------------------

def regularized_logistic_regression_example():
    data = np.loadtxt("ex2data2.txt", delimiter=',')

    # test scores
    X1 = np.vstack(data[:,0])
    X2 = np.vstack(data[:,1])

    X = map_feature(X1, X2)

    m, n = X.shape

    y = np.transpose([data[:,2]])

    theta = np.zeros((n, 1))

    reg_param = 0.1

    # optimize theta using the BFGS algorithm
    res = minimize(fun=cost_function, x0=np.zeros((n, 1)), method="SLSQP", jac=cost_gradient, args=(X, y, reg_param), options={"disp": True})

    # resulting values
    theta = np.vstack(res.x)

    # separate data in order to plot accepted and rejected with different markers
    accepted = [[], []]
    rejected = [[], []]
    for i in range(m):
        if y[i] == 1.0:
            accepted[0].append(X[i][1])
            accepted[1].append(X[i][2])
        elif y[i] == 0.0:
            rejected[0].append(X[i][1])
            rejected[1].append(X[i][2])

    u = np.linspace(X[:,1].min(), X[:,2].max(), 75)
    v = np.linspace(X[:,1].min(), X[:,2].max(), 75)

    z = np.zeros((u.size, v.size))
    for i in range(u.size):
        for j in range(v.size):
            z[i][j] = map_feature(np.array([u[i]]), np.array([v[j]])) @ theta

    # plot data
    fig = plt.figure()
    ax = fig.gca()
    plt.contour(u, v, z, 0, colors='b', linewidths=3)
    ax.scatter(accepted[0], accepted[1], c='k', marker='+', s=225, label="y = 1")
    ax.scatter(rejected[0], rejected[1], c='g', marker='x', s=169, label="y = 0")
    ax.set_xlabel("Microchip Test 1")
    ax.set_ylabel("Microchip Test 2")
    ax.legend()
    ax.grid(False)
    plt.show()


if __name__ == "__main__":
    regularized_logistic_regression_example()

#------------------------------------------------------------------------------------
