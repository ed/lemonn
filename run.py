import numpy as np


# activation function is sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))


# derivative of sigmoid function
def sigmoid_p(x):
    return sigmoid(x)*(1-sigmoid(x))

class NN(object):
    def __init__(self):
        self.alpha = 0.1
        self.hiddensize = 32
        self.inputsize = 4
        # initialize weights and bias to a mean of 0
        self.w0 = 2*np.random.random((self.inputsize,self.hiddensize)) - 1
        self.w1 = 2*np.random.random((self.hiddensize,1)) - 1
        self.b = 2*np.random.random((self.inputsize,1)) - 1


    def _forward_pass(self, X):
        l1 = sigmoid(np.dot(X, self.w0) + self.b)
        l2 = sigmoid(np.dot(l1, self.w1) + self.b)
        return l1, l2


    def _backprop(self, X, y):
        # set hidden and output layer
        h, o = self._forward_pass(X)
        # output layer error equals output-output layer
        o_e = y-o
        # output error delta is equal to the error * the derivative of the
        # logicistic function(sigmoid) of output layer array
        o_d = o_e*sigmoid_p(o)
        # hidden layer error equals matrix multiplication of output delta and
        # transposition of w1
        h_e = np.dot(o_d, w1.T)
        # hidden error delta is equal to the error * the derivative of the
        # logicistic function(sigmoid) of hidden layer array
        h_d = h_e*sigmoid_p(h)
        # update bias and weights
        self.b = o_d
        self.w1 += self.alpha * np.dot(h.T, o_d)
        self.w1 += self.alpha * np.dot(o.T, h_d)

    def train(self, inputs):



