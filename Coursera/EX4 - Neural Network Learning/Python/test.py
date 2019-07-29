import numpy as np
import os
from mnist_web import mnist
train_images, train_labels, test_images, test_labels = mnist(path=os.path.join(os.getcwd(), "mnist"))

def init_weights(L_in, L_out, epsilon_init=0.1):
    weights = np.random.random((L_out, L_in + 1))
    weights *= (2 * epsilon_init)
    weights -= epsilon_init
    return weights

def relu_gradient(z):
    z[z >= 0] = 1
    z[z < 0] = 0
    return z

from scipy.special import softmax
from scipy.optimize import minimize


def cost(Theta, images, labels, reg_param):
    print("Entered cost")
    Theta1_ = Theta[:78500].reshape((100, 785))
    Theta2_ = Theta[78500:78500+10100].reshape((100, 101))
    Theta3_ = Theta[78500+10100:].reshape((10, 101))
    
    log_epsilon = 1e-6
    
    batch_size = 60000
    
    cost = 0
    
    for i in range(batch_size):
        x = images[i]
        y = labels[i]
        
        a1 = np.insert(x, 0, 1)

        z2 = Theta1 @ a1
        # relu activation
        a2 = z2
        a2[a2 < 0] = 0
        a2 = np.insert(a2, 0, 1)
        
        z3 = Theta2 @ a2
        # relu activation
        a3 = z3
        a3[a3 < 0] = 0
        a3 = np.insert(a3, 0, 1)
                
        z4 = Theta3 @ a3
        # softmax activation
        h_theta = softmax(z4)
        
        cost -= sum((np.multiply(y, np.log(h_theta + log_epsilon))) + (np.multiply((1 - y), np.log(1 - h_theta + log_epsilon))))

    cost /= batch_size
    cost += (sum(Theta1[:,1:].flatten() ** 2) + sum(Theta2[:,1:].flatten()) + sum(Theta3[:,1:].flatten())) * (reg_param / (3 * batch_size))
    
    print("Exiting cost")
    return cost

def gradient(Theta, images, labels, reg_param):
    print("Entered gradient")
    Theta1_ = Theta[:78500].reshape((100, 785))
    Theta2_ = Theta[78500:78500+10100].reshape((100, 101))
    Theta3_ = Theta[78500+10100:].reshape((10, 101))
    
    Theta1_Grad = np.zeros(Theta1.shape)
    Theta2_Grad = np.zeros(Theta2.shape)
    Theta3_Grad = np.zeros(Theta3.shape)
        
    batch_size = 60000
        
    for i in range(batch_size):
        x = images[i]
        y = labels[i]
        
        a1 = np.insert(x, 0, 1)

        z2 = Theta1 @ a1
        # relu activation
        a2 = z2
        a2[a2 < 0] = 0
        a2 = np.insert(a2, 0, 1)
        
        z3 = Theta2 @ a2
        # relu activation
        a3 = z3
        a3[a3 < 0] = 0
        a3 = np.insert(a3, 0, 1)
                
        z4 = Theta3 @ a3
        # softmax activation
        h_theta = softmax(z4)
        
        delta4 = h_theta - y
        delta3 = np.multiply((delta4 @ Theta3)[1:], relu_gradient(z3))
        delta2 = np.multiply((delta3 @ Theta2)[1:], relu_gradient(z2))
        
        Theta1_Grad += delta2.reshape((100, 1)) @ a1.reshape((1, 785))
        Theta2_Grad += delta3.reshape((100, 1)) @ a2.reshape((1, 101))
        Theta3_Grad += delta4.reshape((10, 1)) @ a3.reshape((1, 101))
        
    Theta1_Grad /= batch_size
    Theta2_Grad /= batch_size
    Theta3_Grad /= batch_size

    Theta1_Grad[:,1:] += (reg_param / batch_size) * Theta1[:,1:]
    Theta2_Grad[:,1:] += (reg_param / batch_size) * Theta2[:,1:]
    Theta3_Grad[:,1:] += (reg_param / batch_size) * Theta3[:,1:]
    
    gradient = np.hstack((Theta1_Grad.flatten(), Theta2_Grad.flatten(), Theta3_Grad.flatten())).reshape((89610, 1))
    
    print("Exiting gradient")
    return gradient

Theta1 = init_weights(784, 100)
Theta2 = init_weights(100, 100)
Theta3 = init_weights(100, 10)

Theta = np.hstack((Theta1.flatten(), Theta2.flatten(), Theta3.flatten())).reshape((89610, 1))

reg_param = 1.0
    

# optimize theta using the BFGS algorithm
res = minimize(fun=cost, x0=Theta, method="BFGS", jac=gradient, args=(train_images, train_labels, reg_param), options={"disp": True})

Theta = res.x.flatten()

Theta1_ = Theta[:78500].reshape(Theta1.shape)
Theta2_ = Theta[78500:78500+10100].reshape(Theta2.shape)
Theta3_ = Theta[78500+10100:].reshape(Theta3.shape)

correct_count = 0
incorrect_indices = []
for i in range(10000):
    x = test_images[i]
    y = test_labels[i]

    a1 = np.insert(x, 0, 1)

    z2 = Theta1 @ a1
    # relu activation
    a2 = z2
    a2[a2 < 0] = 0
    a2 = np.insert(a2, 0, 1)

    z3 = Theta2 @ a2
    # relu activation
    a3 = z3
    a3[a3 < 0] = 0
    a3 = np.insert(a3, 0, 1)

    z4 = Theta3 @ a3
    # softmax activation
    h_theta = softmax(z4)
    
    if np.argmax(h_theta) == np.argmax(y):
        correct_count += 1
    else:
#         print(np.argmax(h_theta), np.argmax(y))
        incorrect_indices.append(i)

print(correct_count / 10000)
print(len(incorrect_indices))