#!/usr/bin/env python
#---------------------------------------------------------------------
# Author: Will Fenton
# Date: July 24 2019
# 
# Logistic Regression 
# Used for Binary Classification
#---------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#---------------------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost_function(theta, X, y):
    m, n = X.shape
    eps = 1e-8

    h = sigmoid(X @ theta.T)
    cost = ((-y.T  @ np.log(h + eps) - (1 - y).T @ np.log(1 - h + eps)) / m)[0]

    return cost


def cost_gradient(theta, X, y):
    m, n = X.shape

    h = np.vstack(sigmoid(X @ theta.T))
    gradient = (X.T @ (h - y)) / m

    return gradient.T[0]

#---------------------------------------------------------------------

def logistic_regression_example():
    data = np.loadtxt("ex2data1.txt", delimiter=',')

    # test scores
    X = data[:,:2]
    # values
    y = np.transpose([data[:,2]])

    # m is the number of datapoints, n is the number of features
    m, n = X.shape

    # add a leftmost column of ones to X
    X = np.hstack((np.ones((m, 1)), X))

    # optimize theta using the BFGS algorithm
    res = minimize(fun=cost_function, x0=np.zeros((n + 1)), method="BFGS", jac=cost_gradient, args=(X, y), options={"disp": True})

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

    plot_x = np.linspace(0, 100, 2)
    plot_y = (-1 / theta[2]) * (theta[2] * plot_x + theta[0])

    # plot data
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(plot_x, plot_y, "m-", label="Decision Boundary", linewidth=2)
    ax.scatter(accepted[0], accepted[1], c='b', marker='x', s=121, label="Accepted")
    ax.scatter(rejected[0], rejected[1], c='y', marker='x', s=121, label="Rejected")
    ax.set_xlabel("Exam 1 Score")
    ax.set_xlim(0, 100)
    ax.set_ylabel("Exam 2 Score")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    logistic_regression_example()

#---------------------------------------------------------------------
