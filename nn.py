import numpy as np
import simplejson as json
import utils
import math


class NN(object):
    def __init__(self, learning_rate, hidden_size, input_size, output_size):
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size


    def _forward(self, X):
        hidden = utils.sigmoid(X.dot(self.w) + self.b)
        output = utils.sigmoid(hidden.dot(self.w2) + self.b2)
        return hidden, output


    def _backwards(self, X, y, i):
        # output layer error equals output-output layer
        # output error delta is equal to the error * the derivative of the
        # logicistic function(sigmoid) of output layer array
        # hidden layer error equals matrix multiplication of output delta and
        # transpose of w2
        # hidden error delta is equal to the error * the derivative of the
        # logicistic function(sigmoid) of hidden layer array
        # update bias and weights
        hidden, output  = self._forward(X)
        d_o = y - output
        if (i % 10000) == 0:
            print(np.mean(np.abs(d_o)))
        d_o = d_o * utils.dsigmoid(output)
        d_h = np.dot(d_o, self.w2.T) * utils.dsigmoid(hidden)
        self.w2 += self.learning_rate * np.dot(hidden.T, d_o)
        self.b2 += self.learning_rate * np.sum(d_o, axis=0, keepdims=True)
        self.w += self.learning_rate * np.dot(X.T, d_h)
        self.b += self.learning_rate * np.sum(d_h, axis=0, keepdims=True)


    def cross_validate(self, x, y):
        scores = []
        raws = []
        n = math.ceil(len(x)/5)
        x_chunks = [x[k:k+n] for k in range(0, len(x), n)]
        y_chunks = [y[k:k+n] for k in range(0, len(y), n)]
        for i in range(5):
            xtrain = x_chunks[(i+1) % 5]
            ytrain = y_chunks[(i+1) % 5]
            for j in range(3):
                x_train = np.concatenate((x_chunks[(i+j+2) % 5], xtrain))
                y_train = np.concatenate((y_chunks[(i+j+2) % 5], ytrain))
            self.train(x_train, y_train)
            p = self.predict(x_chunks[i])
            scores.append(utils.score(y_chunks[i], p))
        return scores


    def train(self,x,y):
        self.w = 0.01 * np.random.randn(self.input_size, self.hidden_size)
        self.b = np.zeros((1, self.hidden_size))
        self.w2 = 0.01 * np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        for i in range(50000):
            self._backwards(x, y, i)


    def predict(self, x):
        _,a = self._forward(x)
        return a


    def save(self,filename):
        data = {"input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "w": json.dumps(self.w.tolist()),
                "w2": json.dumps(self.w2.tolist()),
                "b": json.dumps(self.b.tolist()),
                "b2": json.dumps(self.b2.tolist()),
                "learning_rate": self.learning_rate}
        with open('outputs/'+filename, "w") as f:
            json.dump(data, f)
