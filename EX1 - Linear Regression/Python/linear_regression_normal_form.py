#!/usr/bin/env python
#---------------------------------------------------------------------
# Author: Will Fenton
# Date: July 23 2019
# 
# Linear regression with multiple variables
# Using the closed form equation to determine the coefficients
#---------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------------------------------------

# uses the following property to exactly determine theta: 
# theta = inv(X' * X) * X' * y
def normal_equation(X, y):
    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
    return theta

#---------------------------------------------------------------------

def multivariable_example():
    data = np.loadtxt("ex1data2.txt", delimiter=',')

    # features
    X = data[:,:2]
    # values
    y = np.transpose([data[:,2]])

    theta = normal_equation(X, y)

    house = np.array([
        [1],
        [1650],
        [3]
    ])

    # example of predicting the price of a 1650 sqft, 3 bdrm house
    predicted_value = theta.T.dot(house)[0][0]
    print(predicted_value)


def single_example():
    data = np.loadtxt("ex1data1.txt", delimiter=',')

    # features
    X = np.transpose([data[:,0]])
    # values
    y = np.transpose([data[:,1]])

    theta = normal_equation(X, y)

    y_intercept = theta[0]
    slope = theta[1]

    # plot points and line of best fit
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(X, y, c='r', marker='x')
    x_axis = np.linspace(X.min(), X.max(), 2)
    line = y_intercept + (slope * x_axis)
    ax.plot(x_axis, line, "b-")
    plt.show()


def graduate_admission():
    # data = np.loadtxt("admission_predict.csv", delimiter=',', skiprows=1)

    # X = data[:,1:8]
    # y = np.transpose([data[:,8]])

    # theta = normal_equation(X, y)

    # student = np.array([
    #     [1],
    #     [325],
    #     [109],
    #     [3],
    #     [3.5],
    #     [3.4],
    #     [9.5],
    #     [1]
    # ])

    # # example of predicting the price of a 1650 sqft, 3 bdrm house
    # predicted_admission_probability = theta.T.dot(student)[0][0]
    # print(predicted_admission_probability)

    data = np.loadtxt("admission_predict.csv", delimiter=',', skiprows=1)

    X = np.transpose([data[:,6]])
    y = np.transpose([data[:,8]])

    theta = normal_equation(X, y)

    y_intercept = theta[0]
    slope = theta[1]

    # plot points and line of best fit
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(X, y, c='r', marker='x')
    x_axis = np.linspace(X.min(), X.max(), 2)
    line = y_intercept + (slope * x_axis)
    ax.plot(x_axis, line, "b-")
    ax.set_xlabel("CGPA")
    ax.set_ylabel("Probability of Admission")
    plt.show()


if __name__ == "__main__":
    # single_example()
    # multivariable_example()
    graduate_admission()

#---------------------------------------------------------------------
