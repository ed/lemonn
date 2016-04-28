import numpy as np


# activation function is sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))


# derivative of sigmoid function
def sigmoid_p(x):
    return sigmoid(x)*(1-sigmoid(x))

class NN(object):
    def __init__(self):
        self.alpha = .01
        self.hiddensize = 8 
        self.inputsize = 3
        # initialize weights and bias to a mean of 0
        self.w0 = 2*np.random.random((self.inputsize+1,self.hiddensize)) - 1
        self.w1 = 2*np.random.random((self.hiddensize,1)) - 1
        # self.b1 = np.zeros((1, self.hiddensize))
        # self.b2 = np.zeros((1, self.inputsize))


    def _forward_pass(self, X):
        a = []
        a.append(sigmoid(np.dot(X, self.w0)))
        [np.insert(i,0,1) for i in a] 
        a.append(sigmoid(np.dot(a[0], self.w1)))
        # a.append(sigmoid(np.dot(X, self.w0)))
        # a.append(sigmoid(np.dot(a[0], self.w1)))
        return a


    def _backprop(self, X, y, j):
        # set hidden and output layer
        a = self._forward_pass(X)
        h,o = a[0], a[1]
        # output layer error equals output-output layer
        o_e = y-o
        if (j % 10000) == 0:
            print(np.mean(np.abs(o_e)))
        # output error delta is equal to the error * the derivative of the
        # logicistic function(sigmoid) of output layer array
        o_d = o_e*sigmoid_p(o)
        # hidden layer error equals matrix multiplication of output delta and
        # transposition of w1
        h_e = np.dot(o_d, self.w1.T)
        # hidden error delta is equal to the error * the derivative of the
        # logicistic function(sigmoid) of hidden layer array
        h_d = h_e*sigmoid_p(h)
        # update bias and weights
        # self.b1 += self.alpha * np.sum(o_d, axis=0)
        # self.b2 += self.alpha * np.sum(h_d, axis=0)
        self.w1 += self.alpha * np.dot(h.T, o_d)
        self.w0 += self.alpha * np.dot(X.T, h_d)

    def train(self):
        # inputs are from file 0017_02_55_40_1_300to600_kuhn_dbl_bs_cosuvd.csv
        # inputs are a tuple ([TVi, TVe, e-time], [dbl, bs, co, su])
        # for now it's just 0 or 1, normal or flow async
        inputs = [ ([563, 566, 5.2], 0), ([492, 523, 6.54], 0), ([487, 602, 7.08], 0), ([553,589,6.22], 0), ([72, 209, 3.02], 1), ([109, 12, 0.18], 1), ([652, 691, 4.18], 0),([300, 596, 4.66], 0),([211, 28, 0.24], 1),([802, 687, 1.38], 1),([875, 550, 2.42], 1) ]
        x_input = [x[0] for x in inputs]
        [x.insert(0,1) for x in x_input]
        x_input = np.array(x_input)
        y_input = [[x[1]] for x in inputs]
        for i in range(100000):
            self._backprop(x_input, y_input, i)

    def predict(self, x):
        a = self._forward_pass(x)
        return a[-1]

def main():
    nn = NN()
    nn.train()
    print(nn.predict([1,527,543,4.72]))
    print(nn.predict([1,166,360,1.72]))
    print(nn.predict([1,205,422,2.44]))
    print(nn.predict([1,308, 6, 0.18]))
    print(nn.predict([1,167, 524, 1.94]))

if __name__ == '__main__':
    main()
