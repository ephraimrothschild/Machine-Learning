__author__ = 'Ephraim'
import numpy as np
import io
import random

# This code represents the Perceptron algorithm for determining if a square is tall or wide.
# The data was taken from http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/RectanglesData
# data label will be 0 if rectangle is tall, 1 if it is wide
# The last column in each line is the label

class Perceptron:
    dim = 784
    b = 0
    w = []
    for num in range(0,dim):
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

    def start(self):
        print("Started Training...")
        training_file = open("rectangles_train.amat", "r")
        training_data = np.array(np.loadtxt(training_file))
        for n in range(0, 1000):
            random.shuffle(training_data)
            for data in training_data:
                training_array = data[:-1]
                training_label = (2*data[-1])-1
                self.train(training_array, training_label)
        print("Training Complete!")

        print("Started Testing...")
        test_file = open("C:/Users/Ephraim/Downloads/rectangles/rectangles_test.amat", "r")
        text_lines = test_file.readlines()
        lines = []
        for num in range(0, 50):
            lines.append(np.loadtxt(io.StringIO(text_lines[num])))

        true = 0
        false = 0
        wide = 0
        real_tall = 0
        real_wide = 0
        false_tall = 0
        false_wide = 0
        tall = 0
        for num in range(0, 1000):
            line = lines[num]
            result = self.predict(line[:-1])
            label = 2*line[-1] - 1
            if label == -1: real_tall += 1
            else: real_wide +=1
            if result > 0: wide += 1
            else: tall += 1
            if result * label > 0:
                true += 1
            else:
                false += 1
                if label == -1: false_tall += 1
                else: false_wide += 1

        print("Number of Real Tall Rectangles:  " + real_tall.__str__())
        print("Number of Real Wide Rectangles:  " + real_wide.__str__())
        print("Number of Estimated Tall Rectangles:  " + tall.__str__())
        print("Number of Estimated Wide Rectangles:  " + wide.__str__())
        print("Number of False Tall Rectangles:  " + false_tall.__str__())
        print("Number of False Wide Rectangles:  " + false_wide.__str__())
        print("Number of Accurate Estimates:  " + true.__str__())
        print("Number of Errors:              " + false.__str__())
Perceptron().start()

