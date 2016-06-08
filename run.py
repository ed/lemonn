import numpy as np
import math
import simplejson
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# activation function is sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))


# derivative of sigmoid function
def sigmoid_p(x):
    return x*(1-x)


class NN(object):
    def __init__(self):
        self.alpha = .01
        self.hiddensize = 4
        self.inputsize = 4
        # initialize weights and bias to a mean of 0
        self.w0 = 2*np.random.random((self.inputsize+1,self.hiddensize)) - 1
        self.w1 = 2*np.random.random((self.hiddensize+1,1)) - 1
        # self.b1 = np.zeros((1, self.hiddensize))
        # self.b2 = np.zeros((1, self.inputsize))


    def _forward_pass(self, X):
        a = []
        # try:
        a.append(sigmoid(np.dot(X, self.w0)))
            # add bias to outer layer
        # a[0] = np.array([np.insert(i,0,1) for i in a[0]])
        a[0] = np.array([np.append(i,1) for i in a[0]])
        # except ValueError:
            # print(X)
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
        h_d = [i[:-1] for i in h_d]
        # update bias and weights
        # self.b1 += self.alpha * np.sum(o_d, axis=0)
        # self.b2 += self.alpha * np.sum(h_d, axis=0)
        self.w1 += self.alpha * np.dot(h.T, o_d)
        self.w0 += self.alpha * np.dot(X.T, h_d)


    def _get_data(self,filename):
        # right now this is written for SIMPS brooks annotation, will
        # generalize later
        a = np.genfromtxt(filename, delimiter=',')
        b = []
        ind = []
        b_t = a[:,[4,5,6]]
        for i, item in enumerate(b_t):
            if sum(item) == 0:
                b.append(a[i])
                ind.append(i)
        a = np.delete(a,ind, axis=0)
        a = np.delete(a, np.s_[-3:], axis=1)
        b = np.delete(b, np.s_[-3:], axis=1)
        b_l = [(list(x), 0) for x in b]
        a_l = [(list(x), 1) for x in a]
        inputs = np.concatenate((a_l, b_l))
        inputs = list(tuple(x) for x in inputs)
        np.random.shuffle(inputs)
        return inputs

    def _get_simp(self):
        a = np.genfromtxt(filename, delimiter=',')
        vc = a[np.where(a[:, -2])]
        vc = np.delete(vc, np.s_[-2:], axis=1)
        vc_l = [(list(x), 0) for x in vc]
        ps = a[np.where(a[:, -1])]
        ps = np.delete(ps, np.s_[-2:], axis=1)
        ps_l = [(list(x), 1) for x in ps]
        inputs = np.concatenate((vc_l, ps_l))
        inputs = list(tuple(x) for x in inputs)
        np.random.shuffle(inputs)
        return inputs

    def _accuracy(self,y,o):
        print(y,o)
        tp = fp = 0
        roc_x = roc_y = []
        mins = min(o)
        maxs = max(o)
        thrs = np.linspace(mins, maxs, 30)
        n = 0
        for i in y:
            n = n+i[0]
        p = len(y) - n
        for (i,T) in enumerate(thrs):
            for i in range(len(y)):
                if (o[i] > T):
                    if (y[i][0] == 1):
                        tp = tp + 1
                    if (y[i][0]==0):
                        fp = fp + 1
            roc_x.append(fp/float(n))
            roc_y.append(tp/float(p))
        tp = fp = 0
        plt.scatter(roc_x, roc_y)
        plt.show()
        # se = tp/(tp+fn)
        # sp = tn/(tn+fp)
        # return se, sp
        #

    def _preprocess(self):
        # inputs are from file 0017_02_55_40_1_300to600_kuhn_dbl_bs_cosuvd.csv
        # inputs are a tuple ([TVi, TVe, e-time], [dbl, bs, co, su])
        # for now it's just 0 or 1, normal or breath async
        # inputs = [ ([563, 566, 5.2], 0), ([492, 523, 6.54], 0), ([487, 602, 7.08], 0), ([553,589,6.22], 0), ([72, 209, 3.02], 1), ([109, 12, 0.18], 1), ([652, 691, 4.18], 0),([300, 596, 4.66], 0),([211, 28, 0.24], 1),([802, 687, 1.38], 1),([875, 550, 2.42], 1) ]
        # #
        inputs = self._get_data("test.csv")
        # inputs = self.get_data(simpinputs.csv)
        # just take x_inputs in the tuple
        x_input = [x[0] for x in inputs]
        # insert a 1 in front of all x_inputs to act as bias vector
        [x.append(1) for x in x_input]
        # move x_input to a numpy array
        x_input = np.array(x_input)
        # isolate the y variables from the tuple
        y_input = [[x[1]] for x in inputs]
        # print(x_input, y_input)
        return x_input, y_input


    def _cross_validate(self):
        x, y = self._preprocess()
        print(x.shape)
        n = math.ceil(len(x)/5)
        x_chunks = [x[k:k+n] for k in range(0, len(x), n)]
        y_chunks = [y[k:k+n] for k in range(0, len(y), n)]
        for i in range(5):
            xtrain = x_chunks[(i+1) % 5]
            ytrain = y_chunks[(i+1) % 5]
            for j in range(3):
                xtrain = np.concatenate((x_chunks[(i+j+2) % 5], xtrain))
                ytrain = np.concatenate((y_chunks[(i+j+2) % 5], ytrain))
            self.train(xtrain, ytrain)
            p = self.predict(x_chunks[i])
            self._accuracy(y_chunks[i], p)

    def train(self,x,y):
        # loop for backpropagation
        self.w0 = 2*np.random.random((self.inputsize + 1,self.hiddensize)) - 1
        self.w1 = 2*np.random.random((self.hiddensize + 1,1)) - 1
        print(x.shape, y.shape)
        for i in range(1):
            self._backprop(x, y, i)

    def predict(self, x):
        a = self._forward_pass(x)
        return a[-1]


    def save(self,filename):
        data = {"inputs": self.inputsize,
                "hidden": self.hiddensize,
                "w0": self.w0,
                "w1": self.w1,
                "alpha": self.alpha}
        f = open(filename, "w")
        simplejson.dump(data, f)
        f.close()


def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    n = NN()
    n.inputsize = data["inputs"]
    n.hiddensize = data["hidden"]
    n.w0 = data["w0"]
    n.w1 = data["w1"]
    n.alpha = data["alpha"]
    return n 


def main():
    nn = NN()
    nn._cross_validate()
    # x,y = nn._preprocess()
    # nn.train(x,y)
    # print(nn.predict([[353,308,1.74,.87,1]]))
    # print(nn.predict([[367,350,2.28,.96,1]]))
    # print(nn.predict([[527,543,4.98,1]])) # 0
    # print(nn.predict([[166,360,1.72,1]])) # 0
    # print(nn.predict([[205,422,2.44,1]])) # 0
    # print(nn.predict([[308, 6, 0.18,1]])) # 1
    # print(nn.predict([[167, 524, 1.94,1]])) # 1

if __name__ == '__main__':
    main()
