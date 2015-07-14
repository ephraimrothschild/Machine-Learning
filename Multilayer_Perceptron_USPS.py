__author__ = 'Ephraim'
import numpy as np
import io
import random
import itertools
from operator import itemgetter

# The data was taken from the Zip Code data at:
#    http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html

class Perceptron:
    def __init__(self, dim=256):
        self.dim = dim
        self.b = 0
        self.w = [0.0]*dim

    def auto_init(self, number_to_classify):
        self.number_to_classify = number_to_classify
        self.train()
        return self

    def train_line(self, x, y):
        a = self.predict_for_training(x)
        if a * y < 1:
            for d in range(0, self.dim):
                self.w[d] += (y * x[d])*0.01
            self.b += y
        return 0

    def predict_for_training(self,x):
        # print(self.w)
        # print(x)
        a = np.dot(self.w, x) + self.b
        return a

    def predict(self, x):
        return np.sign(self.predict_for_training(x))

    def train(self):
        print("Started Training...")
        training_file = open("usps.train", "r")
        raw_training_data = np.loadtxt(training_file).tolist()
        num_data = []
        not_num_data = []
        training_data = []
        for data in raw_training_data:
            training_array = data[1:]
            training_label = data[0]
            if training_label == self.number_to_classify:
                training_array.append(1)
                num_data.append(training_array)
            else:
                training_array.append(-1)
                not_num_data.append(training_array)
        for num in range(0,len(num_data)):
            training_data.append(num_data[num])
            training_data.append(not_num_data[num])
        for n in range(0, 10):
            random.shuffle(training_data)
            for data in training_data:
                training_array = data[:-1]
                training_label = data[-1]
                self.train_line(training_array, training_label)
        return self

class MulticlassPerceptron:
    def __init__(self, depth=0):
        self.perceptrons = {}
        self.lower_bound = 0
        self.upper_bound = 0
        self.TL_Perceptron = {}

    def train(self, low_bound, up_bound):
        self.lower_bound = low_bound
        self.upper_bound = up_bound
        for n in range(self.lower_bound, self.upper_bound):
            self.perceptrons[n] = Perceptron().auto_init(n)

    def predict(self, x):
        probability = {}
        for n in range(self.lower_bound, self.upper_bound):
            prediction_certainty = self.perceptrons[n].predict(x)
            if prediction_certainty > 0:
                probability[n] = 1
            else:
                probability[n] = prediction_certainty
            return sorted(probability.items(), key=itemgetter(1), reverse=True)[0][0]

    def prediction_array(self, x):
        probability = {}
        for n in range(self.lower_bound, self.upper_bound):
            prediction_certainty = self.perceptrons[n].predict(x)
            if prediction_certainty > 0:
                probability[n] = 1
            else:
                probability[n] = prediction_certainty
            sorted_probability = sorted(probability.items(), key=itemgetter(0))
        return np.array([touple[1] for touple in sorted_probability])

class LayeredPerceptrion:
    def __init__(self):
        self.perceptrons = {}
        self.lower_bound = 0
        self.upper_bound = 0
        self.multi_class = MulticlassPerceptron()
        self.multi_class.train(0, 10)

    def train(self, low_bound, up_bound):
        self.lower_bound = low_bound
        self.upper_bound = up_bound
        training_file = open("usps.train", "r")
        raw_training_data = np.loadtxt(training_file).tolist()
        for n in range(self.lower_bound, self.upper_bound):
            self.perceptrons[n] = Perceptron(10)
            num_data = []
            not_num_data = []
            training_data = []
            for data in raw_training_data:
                training_array = self.multi_class.prediction_array(data[1:])
                training_label = data[0]
                if training_label == n:
                    training_array = np.append(training_array, 1)
                    num_data.append(training_array)
                else:
                    training_array = np.append(training_array, -1)
                    not_num_data.append(training_array)
            for num in range(0, len(num_data)):
                training_data.append(num_data[num])
                training_data.append(not_num_data[num])
            for data in training_data:
                self.perceptrons[n].train_line(data[:-1], data[-1])

    def predict(self, x):
        probability = {}
        for n in range(self.lower_bound, self.upper_bound):
            prediction_certainty = self.perceptrons[n].predict(self.multi_class.prediction_array(x))
            if prediction_certainty > 0:
                probability[n] = 1
            else:
                probability[n] = prediction_certainty
        sorted_probability = sorted(probability.items(), key=itemgetter(1), reverse=True)
        return sorted_probability[0][0]



def start():
    print("Started Training...")
    multi_class = LayeredPerceptrion()
    multi_class.train(0, 10)
    print("Training Complete!")
    print("Started Testing...")
    test(multi_class)
    print(multi_class.lower_bound)
    print(multi_class.upper_bound)
    for p in multi_class.perceptrons.items():
        print(p[1].w)
    print(multi_class.perceptrons[0].w == multi_class.perceptrons[1].w)

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
        result = multiclass.predict(line[1:])
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
# for number in range(0, 3):
#     print("Attempting to learn then number " + number.__str__())
#     Perceptron(number).start2()
#     print()