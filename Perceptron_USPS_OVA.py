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

    def __init__(self, number_to_classify):
        self.number_to_classify = number_to_classify
        self.b = 0
        self.w = []
        for num in range(0, self.dim):
            self.w.append(0)
        self.train()

    def train_line(self, x, y):
        a = self.predict_for_training(x)
        if a * y < 1:
            for d in range(0, self.dim):
                self.w[d] += (y * x[d])*0.01
            self.b += y
        return 0

    def predict_for_training(self,x):
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
        # fig = plt.figure()
        # weight_matrix = np.reshape(np.array(self.w), (16, 16))
        # plt.contourf(weight_matrix)
        # plt.show(block=False)
        # im.set_array(weight_matrix)
        # fig.canvas.draw()
        for n in range(0, 10):
            random.shuffle(training_data)
            for data in training_data:
                training_array = data[:-1]
                training_label = data[-1]
                self.train_line(training_array, training_label)
                # weight_matrix = np.reshape(np.array(self.w), (16, 16))
                # plt.clear()
                # plt.contourf(weight_matrix)
                # plt.pause(0.0000001)
            # print(n)
        return self
        # plt.close()

    def start2(self):
        print("Training Complete!")
        print(self.w)
        print("Started Testing...")
        test_file = open("usps.test", "r")
        text_lines = test_file.readlines()
        lines = []
        for num in range(0, 1000):
            lines.append(np.loadtxt(io.StringIO(text_lines[num])))

        true = 0
        false = 0
        is_num = 0
        real_not_num = 0
        real_is_num = 0
        false_not_num = 0
        false_is_num = 0
        not_num = 0
        for num in range(0, 1000):
            line = lines[num]
            result = self.predict(line[1:])
            if line[0] == self.number_to_classify:
                real_is_num += 1
                is_label_num = 1
            else:
                real_not_num += 1
                is_label_num = -1
            if result > 0: is_num += 1
            else: not_num += 1
            if result * is_label_num > 0:
                true += 1
            else:
                false += 1
                if is_label_num == -1: false_not_num += 1
                else: false_is_num += 1
        print("Number of Real "+self.number_to_classify.__str__()+"s:  " + real_is_num.__str__())
        print("Number of Real Non-"+self.number_to_classify.__str__()+"s:  " + real_not_num.__str__())
        print("Number of Estimated "+self.number_to_classify.__str__()+"s:  " + is_num.__str__())
        print("Number of Estimated Non-"+self.number_to_classify.__str__()+"s:  " + not_num.__str__())
        print("Number of False "+self.number_to_classify.__str__()+"s:  " + false_is_num.__str__())
        print("Number of False Non-"+self.number_to_classify.__str__()+"s:  " + false_not_num.__str__())
        print("Number of Accurate Estimates:  " + true.__str__())
        print("Number of Errors:              " + false.__str__())



class MulticlassPerceptron:
    # perceptron_numbers = []

    def __init__(self):
        self.perceptrons = {}
        self.lower_bound = 0
        self.upper_bound = 0

    def train(self, low_bound, up_bound):
        self.lower_bound = low_bound
        self.upper_bound = up_bound
        for n in range(self.lower_bound, self.upper_bound):
            self.perceptrons[n] = Perceptron(n)

    def predict(self, x):
        probability = {}
        for n in range(self.lower_bound, self.upper_bound):
            prediction_certainty = self.perceptrons[n].predict(x)
            # print(n.__str__() + " : " + prediction_certainty.__str__() + "  ,  ", end="")
            if prediction_certainty > 0:
                probability[n] = 1
            else:
                probability[n] = prediction_certainty
            # else:
            #     for prob in range(0, len(probability)):
            #         if prob != self.perceptron_numbers[combination_number]:
            #             probability[prob] += 1 #prediction_certainty
        sorted_probability = sorted(probability.items(), key=itemgetter(1), reverse=True)
        return sorted_probability[0][0]

def start():
    print("Started Training...")
    multi_class = MulticlassPerceptron()
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