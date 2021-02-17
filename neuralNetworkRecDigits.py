import numpy
import scipy.special
import pandas as pd
import matplotlib.pyplot


# %matplotlib inline


class NeuralNetwork:
    # fuc initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set the number of nodes in different layers
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # initialise the weight matrix
        # inputs and hidden layer
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        # hidden and output layer
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # initialise the learning rate
        self.lr = learningrate

        # initialise the activation function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        # covert the inputs into 2-D arrays
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate the signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals after activation
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate the signals into output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculatee the signals after activation/final outputs
        final_outputs = self.activation_function(final_inputs)

        # output layer error
        output_errors = targets - final_outputs
        # hidden layer errors
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights backward propagation
        # update the weight between hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        # update the weight between input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    # test function
    def query(self, inputs_list):
        # covert the inputs into 2-D arrays
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate the signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals after activation
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate the signals into output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals after activation/final outputs
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

        pass


input_nodes = 28 * 28  # the pixels number of a picture
hidden_nodes = 100  # undetermined
output_nodes = 10  # all possibilities of a digit
learning_rate = 0.1  # undetermined

# create a instance of the neural network
digit_recognizer = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the training data CSV file into a list
training_data_file = open("E:/Pycharm/WestTwo/ThirdTerm/train.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network
epochs = 5  # the number of times training the neural network with the same network

for e in range(epochs):
    for record in training_data_list[1:]:
        # split the record by the ','
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
        # create the target output values, all false values are assigned 0.01, true value is assigned 0.99
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        digit_recognizer.train(inputs, targets)
        pass
    pass

# load the test data CSV file into a list
test_data_file = open("E:/Pycharm/WestTwo/ThirdTerm/test.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

# store the predict result
predict = []
# use test function to test the neural network
for record in test_data_list[1:]:
    # split the record by the ','
    all_values = record.split(',')
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values) / 255.0 * 0.99) + 0.01
    outputs = digit_recognizer.query(inputs)
    label = numpy.argmax(outputs)
    predict.append(label)
    pass

order_list = list(range(len(predict)))
df = pd.DataFrame([order_list, predict]).T
df.columns = ["ImageId", "Label"]
df.to_csv('submission.csv', encoding='gbk')
