#!/usr/bin/env python
#---------------------------------------------------------------------
# Author: Will Fenton
# Date: July 23 2019
# 
# Linear regression with multiple variables, using gradient descent
#---------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------------------------------------

# normalize our features to speed up gradient descent
def normalize_features(X):
    m, n = X.shape

    mu = X.mean(axis=0)
    std = X.mean(axis=0)

    for i in range(n):
        X[:,i] = (X[:,i] - mu[i]) / std[i]

    return X, mu, std


# use gradient descent to find the optimal values to mizimise the cost function
# learning_rate determines the step size
#   - too small and it can take too long, too large and it can diverge
# num_iterations is the number of steps
def gradient_descent(X, y, learning_rate=0.1, num_iterations=1000):
    m, n = X.shape

    # the coefficients for our line of best fit
    theta = np.zeros((n, 1))

    cost_history = []

    for i in range(num_iterations):
        predictions = X.dot(theta)
        error = predictions - y
        delta = (X.T.dot(error)) * (learning_rate / m)
        theta -= delta

        cost = mean_squared_error(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history


# mean squared error is our cost function
# we use gradient descent to find the values for theta which minimize this
def mean_squared_error(X, y, theta):
    m, n = X.shape

    predictions = X.dot(theta)
    error = predictions - y
    mse = sum(error ** 2) / (2 * m)

    return mse


# plot the cost history to evaluate gradient descent's performance
# ideally the cost will drop quickly and then level out
def plot_cost(cost_history, learning_rate, num_iterations):
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(cost_history)
    plt.title(f"Gradient Descent - {learning_rate} learning rate - {num_iterations} iterations")
    ax.set_xlabel("# of iterations of gradient descent")
    ax.set_ylabel("cost (mean squared error)")
    plt.show()

#---------------------------------------------------------------------

def multivariable_example():
    data = np.loadtxt("ex1data2.txt", delimiter=',')

    # features
    X = data[:,:2]
    # values
    y = np.transpose([data[:,2]])

    # m is the number of datapoints, n is the number of features
    m, n = X.shape

    X, mu, std = normalize_features(X)

    # add a leftmost column of ones to X
    X = np.hstack((np.ones((m, 1)), X))

    # learning_rate determines the step size
    # too small and it can take too long, too large and it can diverge
    learning_rate = 1.0

    # num_iterations is the number of steps
    num_iterations = 1000

    theta, cost_history = gradient_descent(X, y, learning_rate, num_iterations)

    plot_cost(cost_history, learning_rate, num_iterations)

    # example of predicting the price of a 1650 sqft, 3 bdrm student
    house = np.array([
        [1],
        [(1650 - mu[0]) / std[0]],
        [(3 - mu[1]) / std[1]]
    ])

    predicted_value = theta.T.dot(house)[0][0]
    print(predicted_value)


def single_example():
    data = np.loadtxt("ex1data1.txt", delimiter=',')

    # features
    X = np.transpose([data[:,0]])
    # values
    y = np.transpose([data[:,1]])

    # m is the number of datapoints, n is the number of features
    m, n = X.shape

    X, mu, std = normalize_features(X)

    # add a leftmost column of ones to X
    X = np.hstack((np.ones((m, 1)), X))

    # learning_rate determines the step size
    # too small and it can take too long, too large and it can diverge
    learning_rate = 1.0

    # num_iterations is the number of steps
    num_iterations = 50

    theta, cost_history = gradient_descent(X, y, learning_rate, num_iterations)

    plot_cost(cost_history, learning_rate, num_iterations)


def graduate_admission():
    data = np.loadtxt("admission_predict.csv", delimiter=',', skiprows=1)

    X = data[:,1:8]
    y = np.transpose([data[:,8]])

    # m is the number of datapoints, n is the number of features
    m, n = X.shape

    X, mu, std = normalize_features(X)

    # add a leftmost column of ones to X
    X = np.hstack((np.ones((m, 1)), X))

    # learning_rate determines the step size
    # too small and it can take too long, too large and it can diverge
    learning_rate = 1.5

    # num_iterations is the number of steps
    num_iterations = 1000

    theta, cost_history = gradient_descent(X, y, learning_rate, num_iterations)

    plot_cost(cost_history, learning_rate, num_iterations)

    # example of predicting the price of a 1650 sqft, 3 bdrm student
    student = np.array([
        [1],
        [(325 - mu[0]) / std[0]],
        [(109 - mu[1]) / std[1]],
        [(3 - mu[2]) / std[2]],
        [(3.5 - mu[3]) / std[3]],
        [(3.4 - mu[4]) / std[4]],
        [(9.5 - mu[5]) / std[5]],
        [(1 - mu[6]) / std[6]]
    ])

    predicted_admission_probability = theta.T.dot(student)[0][0]
    print(predicted_admission_probability)


if __name__ == "__main__":
    # single_example()
    # multivariable_example()
    graduate_admission()

#---------------------------------------------------------------------
