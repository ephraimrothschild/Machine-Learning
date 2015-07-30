from operator import itemgetter

class OVA:
    def __init__(self):
        self.perceptron_wrappers = []

    # Adds a perceptron to the OVA Object. The perceptron must have a .predict(x) method
    def add_perceptron(self, perceptron, num_to_classify):
        self.perceptron_wrappers.append(PerceptronWrapper(perceptron, num_to_classify))

    def predict2(self, x):
        probability = {}
        # for n in range(self.lower_bound, self.upper_bound):
        for perceptron_wrapper in self.perceptron_wrappers:

            prediction_certainty = perceptron_wrapper.perceptron.predict(x)
            # print(n.__str__() + " : " + prediction_certainty.__str__() + "  ,  ", end="")
            if prediction_certainty > 0:
                probability[perceptron_wrapper.num_to_classify] = 1
            else:
                probability[perceptron_wrapper.num_to_classify] = prediction_certainty
            # else:
            #     for prob in range(0, len(probability)):
            #         if prob != self.perceptron_numbers[combination_number]:
            #             probability[prob] += 1 #prediction_certainty
        sorted_probability = sorted(probability.items(), key=itemgetter(1), reverse=True)
        return sorted_probability[0][0]

    def predict(self, x):
        probability = {}
        # for n in range(self.lower_bound, self.upper_bound):
        for perceptron_wrapper in self.perceptron_wrappers:
            prediction_certainty = perceptron_wrapper.perceptron.predict(x)
            if prediction_certainty > 0:
                probability[perceptron_wrapper.num_to_classify] = 1
            else:
                probability[perceptron_wrapper.num_to_classify] = prediction_certainty
        sorted_probability = sorted(probability.items(), key=itemgetter(1), reverse=True)
        print(probability)
        return sorted_probability[0][0]

class PerceptronWrapper:
    def __init__(self, perceptron, num_to_classify):
        self.perceptron = perceptron
        self.num_to_classify = num_to_classify