__author__ = 'Ephraim'
import numpy as np
import io
import random
import itertools
from operator import itemgetter

# The data was taken from the Zip Code data at:
#    http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html

class Perceptron:
    dim = 256
    b = 0
    w = []
    for num in range(0, dim):
        w.append(0)

    def train_single_entry(self, x, y):
        a = self.predict(x)
        if a * y < 1:
            for d in range(0, self.dim):
                self.w[d] += (y * x[d])
            self.b += y
        return 0

    def predict(self, x):
        a = np.dot(self.w, x) + self.b
        return a

    def train(self, num_for_positive):
        # print("Started Training...")
        training_file = open("usps.train", "r")
        raw_training_data = np.loadtxt(training_file).tolist()
        num_data = []
        not_num_data = []
        training_data = []
        for data in raw_training_data:
            training_array = data[1:]
            training_label = data[0]
            if training_label == num_for_positive:
                training_array.append(1)
                num_data.append(training_array)
            else:
                training_array.append(-1)
                not_num_data.append(training_array)
        for num in range(0, min(len(num_data), len(not_num_data))):
            training_data.append(num_data[num])
            training_data.append(not_num_data[num])
        for n in range(0, 1):
            # random.shuffle(training_data)
            for data in training_data:
                training_array = data[:-1]
                training_label = data[-1]
                self.train_single_entry(training_array, training_label)
        return self


class MulticlassPerceptron:
    perceptron_numbers = []
    perceptrons = []
    lower_bound = 0
    upper_bound = 0

    def train(self, low_bound, up_bound):
        self.lower_bound = low_bound
        self.upper_bound = up_bound
        number_list = []
        for n in range(self.lower_bound, self.upper_bound):
            number_list.append(n)
        self.perceptron_numbers = number_list
        for number in self.perceptron_numbers:
                self.perceptrons.append(Perceptron().train(number))

    def predict(self, x):
        probability = {}
        for n in range(self.lower_bound, self.upper_bound):
            probability[n] = 0
        for combination_number in range(1, len(self.perceptrons)):
            prediction_certainty = self.perceptrons[combination_number].predict(x)
            if prediction_certainty > 0:
                probability[self.perceptron_numbers[combination_number]] += prediction_certainty
            # else:
            #     for prob in range(0, len(probability)):
            #         if prob != self.perceptron_numbers[combination_number]:
            #             probability[prob] += 1 #prediction_certainty
        sorted_probability = sorted(probability.items(), key=itemgetter(1), reverse=True)
        print(sorted_probability)
        return sorted_probability[0][0]

def start():
    print("Started Training...")
    multi_class = MulticlassPerceptron()
    multi_class.train(0, 10)
    print("Training Complete!")

    print("Started Testing...")
    test(multi_class)

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