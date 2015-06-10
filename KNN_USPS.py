__author__ = 'Ephraim'
import numpy as np
import io
from operator import itemgetter
# The data was taken from the Zip Code data at:
#    http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html

# Train
file = open("usps.train", "r")

training_data = np.array(np.loadtxt(file))

# Distance function
def dist(array1, array2):
    return np.linalg.norm(array1 - array2)

# Pridict the label of a new array
def predict(array):
    # C:\Users\Ephraim\Downloads\rectangles\rectangles_test.amat
    nearbyArrays = []
    for data in training_data:
        training_array = data[1:]
        training_label = data[0]
        nearbyArrays.append([dist(array, training_array), training_label])
    sortedNearbyArrays = sorted(nearbyArrays, key=itemgetter(0))
    votesPerNumber = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for num in range(0, 10):
        votesPerNumber[int(sortedNearbyArrays[num][1])] += 1
    votedOn = -1
    largestVote = -1
    # print(votesPerNumber)
    for num in range(0,9):
        if votesPerNumber[num] > largestVote:
            votedOn = num
            largestVote = votesPerNumber[num]
    # print(votedOn)
    return votedOn

def test():
    test_file = open("usps.test", "r")
    # lines = np.loadtxt(test_file)
    text_lines = test_file.readlines()
    lines = []
    for num in range(0,1000):
        lines.append(np.loadtxt(io.StringIO(text_lines[num])))

    true = 0
    false = 0
    predictions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for num in range(0, 1000):
        line = lines[num]
        result = predict(line[1:])
        # print(result)
        # print(line[-1])
        # print()
        predictions[int(result)] += 1
        if (result == line[0]):
            true += 1
        else:
            false += 1
    for index in range(0,9):
        print("Number of " + index.__str__() + "s predicted:   " + predictions[index].__str__())
    print("Number of Accurate Estimates:  " + true.__str__())
    print("Number of Errors:              " + false.__str__())

test()
