__author__ = 'Ephraim'
import numpy as np
import io
from operator import itemgetter
# The data was taken from http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/RectanglesData
# data label will be 0 if rectangle is tall, 1 if it is wide
# The last column in each line is the label

# Train
file = open("rectangles_train.amat", "r")
# file = open("C:/Users/Ephraim/Downloads/rectangles_images/rectangles_im_test.amat", "r")

training_data = np.array(np.loadtxt(file))

# Distance function
def dist(array1, array2):
    return np.linalg.norm(array1 - array2)

# Pridict the label of a new array
def predict(array):
    # C:\Users\Ephraim\Downloads\rectangles\rectangles_test.amat
    s = []
    for data in training_data:
        training_array = data[:-1]
        training_label = data[-1]
        if training_label < 1:
            training_label = -1
        s.append([dist(array, training_array), training_label])
    S = sorted(s, key=itemgetter(0))
    y = 0
    for num in range(0, 10):
        y += S[num][1]
    if y > 0:
        return 1
    else:
        return 0

def test():
    # test_file = open("C:/Users/Ephraim/Downloads/rectangles_images/rectangles_im_test.amat", "r")
    test_file = open("C:/Users/Ephraim/Downloads/rectangles/rectangles_test.amat", "r")
    # lines = np.loadtxt(test_file)
    text_lines = test_file.readlines()
    lines = []
    for num in range(0,1000):
        lines.append(np.loadtxt(io.StringIO(text_lines[num])))

    true = 0
    false = 0
    wide = 0
    tall = 0
    for num in range(0, 1000):
        line = lines[num]
        result = predict(line[:-1])
        # print(result)
        # print(line[-1])
        # print()
        if result: wide += 1
        else: tall+=1
        if (result == 0 and line[-1] <= 0) or (result == 1 and line[-1] > 0):
            true += 1
        else:
            false += 1
    print("Number of Estimated Tall Rectangles:  " + tall.__str__())
    print("Number of Estimated Wide Rectangles:  " + wide.__str__())
    print("Number of Accurate Estimates:  " + true.__str__())
    print("Number of Errors:              " + false.__str__())

test()
