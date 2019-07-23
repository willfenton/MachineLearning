#!/usr/bin/env python
#------------------------------------------------------------
# Author: Will Fenton
# Date: July 23 2019
# 
# Linear regression with multiple variables
#------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------

def normalize_features(X, n):

    mu = X.mean(axis=0)
    std = X.mean(axis=0)

    for i in range(n):
        X[:,i] = (X[:,i] - mu[i]) / std[i]

    return X, mu, std

def gradient_descent(X, y, learning_rate=0.1, num_iterations=1000):
    m, n = X.shape
    X, mu, std = normalize_features(X, n)
    X = np.hstack((np.ones((m, 1)), X))
    theta = np.zeros((n + 1, 1))

    for i in range(num_iterations):
        predictions = X.dot(theta)
        error = predictions - y
        delta = (X.transpose().dot(error)) * (learning_rate / m)
        theta -= delta

    return theta, mu, std

def gradient_descent_estimate_house_value(theta, mu, std, square_footage, num_bedrooms):
    house = np.array([
        [1],
        [(square_footage - mu[0]) / std[0]],
        [(num_bedrooms - mu[1]) / std[1]]
    ])
    estimated_value = theta.transpose().dot(house)[0][0]
    return estimated_value

def normal_equation(X, y):
    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))
    theta = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(y))
    return theta

def normal_equation_estimate_house_value(theta, square_footage, num_bedrooms):
    house = np.array([
        [1],
        [square_footage],
        [num_bedrooms]
    ])
    estimated_value = theta.transpose().dot(house)[0][0]
    return estimated_value

#------------------------------------------------------------

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    data = np.loadtxt("ex1data2.txt", delimiter=',')

    # features
    X = data[:,:2]

    # values
    y = np.transpose([data[:,2]])

    square_footage = 2050
    num_bedrooms = 3

    normal_theta = normal_equation(X,y)
    normal_prediction = normal_equation_estimate_house_value(normal_theta, square_footage, num_bedrooms)

    learning_rate = 1.0
    num_iterations = 1000

    gradient_descent_theta, mu, std = gradient_descent(X, y, learning_rate, num_iterations)
    gradient_prediction = gradient_descent_estimate_house_value(gradient_descent_theta, mu, std, square_footage, num_bedrooms)

    print(f"\n{square_footage} square feet, {num_bedrooms} bedrooms, estimating price\n")
    print(f"Normal Equation Method:  {normal_prediction}")
    print(f"Gradient Descent Method: {gradient_prediction}")

#------------------------------------------------------------
