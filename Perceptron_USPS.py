__author__ = 'Ephraim'
import numpy as np
import io
import random

# The data was taken from the Zip Code data at:
#    http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html

class Perceptron:
    dim = 256
    b = 0
    w = []
    for num in range(0, dim):
        w.append(0)

    def train(self, x, y):
        a = self.predict_for_training(x)
        if a * y < 1:
            for d in range(0, self.dim):
                self.w[d] += (y * x[d])
            self.b += y
        return 0

    def predict_for_training(self,x):
        a = np.dot(self.w, x) + self.b
        return a

    def predict(self, x):
        return np.sign(self.predict_for_training(x))

    def start(self, number_to_classify):
        print("Started Training...")
        training_file = open("usps.train", "r")
        raw_training_data = np.loadtxt(training_file).tolist()
        num_data = []
        not_num_data = []
        training_data = []
        for data in raw_training_data:
            training_array = data[1:]
            training_label = data[0]
            if training_label == number_to_classify:
                training_array.append(1)
                num_data.append(training_array)
            else:
                training_array.append(-1)
                not_num_data.append(training_array)
        for num in range(0,len(num_data)):
            training_data.append(num_data[num])
            training_data.append(not_num_data[num])
        for n in range(0, 100):
            random.shuffle(training_data)
            for data in training_data:
                training_array = data[:-1]
                training_label = data[-1]
                self.train(training_array, training_label)
        print("Training Complete!")

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
            if line[0] == number_to_classify:
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
        print("Number of Real "+number_to_classify.__str__()+"s:  " + real_is_num.__str__())
        print("Number of Real Non-"+number_to_classify.__str__()+"s:  " + real_not_num.__str__())
        print("Number of Estimated "+number_to_classify.__str__()+"s:  " + is_num.__str__())
        print("Number of Estimated Non-"+number_to_classify.__str__()+"s:  " + not_num.__str__())
        print("Number of False "+number_to_classify.__str__()+"s:  " + false_is_num.__str__())
        print("Number of False Non-"+number_to_classify.__str__()+"s:  " + false_not_num.__str__())
        print("Number of Accurate Estimates:  " + true.__str__())
        print("Number of Errors:              " + false.__str__())

for number in range(0, 10):
    print("Attempting to learn then number " + number.__str__())
    Perceptron().start(number)
    print()

