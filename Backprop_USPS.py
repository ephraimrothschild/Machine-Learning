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
    def __init__(self, k, d, maxitter=10, bias=1):
        self.maxitter = maxitter
        self.d = d
        self.k = k
        self.W = (np.random.rand(k, d)*2 - 1)/10
        self.v = (np.random.rand(k)*2 - 1)/10
        self.bias = bias


    def train(self, inputs, nu):
        for _ in range(0, self.maxitter):
            random.shuffle(inputs)
            all_errors = 0
            for input in inputs:
                wb = np.append(input.x, self.bias)
                x = wb/np.linalg.norm(wb)
                # x = np.append(input.x/np.linalg.norm(input.x), self.bias)
                a = self.W.dot(x)
                h = np.tanh(a)
                y_hat = np.tanh(np.dot(self.v, np.array(h)))
                error = input.y - y_hat
                all_errors = all_errors+error*error
                # print(y_hat, input.y)
                self.v = self.v+nu*error*np.array(h)
                for i in range(0, self.k):
                    self.W[i] = self.W[i] + nu*((error*self.v[i])*(1 - (np.tanh(a[i])**2)))*x
            # print(self.v[0])
            # print()
            # print(all_errors)

    def predict(self, input):
        wb = np.append(input.x, self.bias)
        x = wb/np.linalg.norm(wb)
        # x = np.append(input.x/np.linalg.norm(input.x), self.bias)
        h = np.tanh(self.W.dot(x))
        return np.tanh(np.dot(np.array(h), self.v))




def getInputs(path, num_to_classify):
    print("Started Training...")
    # training_file = open("usps.train", "r")
    training_file = open(path, "r")
    raw_training_data = np.loadtxt(training_file).tolist()
    random.shuffle(raw_training_data)
    num_data = []
    not_num_data = []
    training_data = []
    for data in raw_training_data:
        training_array = data[1:]
        training_label = data[0]
        # training_array.append(1)
        if training_label == num_to_classify:
            num_data.append(Input(training_array, 1))
        else:
            not_num_data.append(Input(training_array, -1))
    for num in range(0,len(num_data)):
        training_data.append(num_data[num])
        training_data.append(not_num_data[num])
    return training_data


def start():
    # print("Started Training...")
    ova = Generic_OVA.OVA()
    for num in range(0, 10):
        training_data = getInputs("usps.train", num)
        backprop = BP(15, 257, 20)
        backprop.train(training_data, 0.2)
        ova.add_perceptron(backprop, num)
    test(ova)
    # training_data = getInputs("usps.train", 2)
    # backprop = BP(10, 256, 10)
    # backprop.train(training_data, 1)
    # test2(backprop, 5)

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
        inVec = line[1:]
        # inVec = np.append(inVec, 1)
        result = multiclass.predict(Input(inVec, line[0]))
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