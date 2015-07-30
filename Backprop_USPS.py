__author__ = 'Ephraim'
import numpy as np
import io
import Generic_OVA
import random

class Input:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = y

class BP:
    def __init__(self, k, d, maxitter):
        if maxitter == None:
            self.maxitter = 1
        else:
            self.maxitter = maxitter
        self.d = d
        self.k = k
        self.W = np.random.rand(k, d)
        self.v = np.random.rand(k)


    def train(self, inputs, en):
        for _ in range(0, self.maxitter):
            random.shuffle(inputs)
            G = np.zeros((self.k, self.d))
            g = np.zeros(self.k)
            for input in inputs:
                a = []
                h = []
                for i in range(0, self.k):
                    ai = np.dot(self.W[i], input.x)
                    a.append(ai)
                    hi = np.tanh(ai)
                    h.append(hi)
                y_hat = np.dot(self.v, np.array(h))
                error = input.y - y_hat
                g = g - error*np.array(h)
                for i in range(0, self.k):
                    G[i] = G[i] - ((error*self.v[i])*(1 - (np.tanh(a[i])**2)))*input.x
            self.W = self.W - en*G
            self.v = self.v - en*g

    def predict(self, input):
        h = []
        for i in range(0, self.k):
            h.append(np.tanh(np.dot(self.W[i], input.x)))
        return np.dot(np.array(h), self.v)




def getInputs(path, num_to_classify):
    print("Started Training...")
    # training_file = open("usps.train", "r")
    training_file = open(path, "r")
    raw_training_data = np.loadtxt(training_file).tolist()
    num_data = []
    not_num_data = []
    training_data = []
    for data in raw_training_data:
        training_array = data[1:]
        training_label = data[0]
        if training_label == num_to_classify:
            num_data.append(Input(training_array, 1))
        else:
            not_num_data.append(Input(training_array, -1))
    for num in range(0,len(num_data)):
        training_data.append(num_data[num])
        training_data.append(not_num_data[num])
    return training_data


def start():
    print("Started Training...")
    ova = Generic_OVA.OVA()
    for num in range(0, 10):
        training_data = getInputs("usps.train", num)
        backprop = BP(10, 256, 10)
        backprop.train(training_data, 1)
        ova.add_perceptron(backprop, num)
    test(ova)
    # training_data = getInputs("usps.train", 2)
    # backprop = BP(10, 256, 10)
    # backprop.train(training_data, 1)
    # test2(backprop, 2)

def test2(network, num):
    test_data = getInputs("usps.test", num)
    true = 0
    false = 0
    numofnum = 0
    numnotnum = 0
    real = 0

    for data in test_data:
        result = network.predict(data)
        if result > 0:
            numofnum = numofnum + 1
        else:
            numnotnum = numnotnum + 1
        if data.y == 1:
            real  = real + 1
        if (result > 0 and data.y > 0) or (result < 0 and data.y < 0):
            true += 1
        else:
            false += 1

    print("Number of " + num.__str__() + "s predicted:   " + numofnum.__str__())
    print("Number of other things predicted:   " + numnotnum.__str__())

    print("                     Vs:  "+ real.__str__() + " real ones")
    print()
    print("Number of Accurate Estimates:  " + true.__str__())
    print("Number of Errors:              " + false.__str__())


def test(multiclass):
    test_file = open("usps.test", "r")
    # lines = np.loadtxt(test_file)
    text_lines = test_file.readlines()
    lines = []
    for num in range(0, 1000):
        lines.append(np.loadtxt(io.StringIO(text_lines[num])))

    true = 0
    false = 0
    predictions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for num in range(0, 1000):
        line = lines[num]
        result = multiclass.predict(Input(line[1:], line[0]))
        predictions[int(result)] += 1
        labels[int(line[0])] += 1
        if result == line[0]:
            true += 1
        else:
            false += 1
    for index in range(0, 10):
        print("Number of " + index.__str__() + "s predicted:   " + predictions[index].__str__())
        print("                     Vs:  "+ labels[index].__str__() + " real ones")
    print()
    print("Number of Accurate Estimates:  " + true.__str__())
    print("Number of Errors:              " + false.__str__())

start()