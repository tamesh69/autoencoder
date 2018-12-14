import numpy as np
import scipy
import pickle

class Net:
    def __init__(self, inputnodes, el1, el2, ml, dl2, dl1, outputnodes, learning_rate):
        self.input_nodes = inputnodes
        self.encoding_nodes1 = el1
        self.encoding_nodes2 = el2
        self.middle_nodes = ml
        self.decoding_nodes2 = dl2
        self.decoding_nodes1 = dl1
        self.output_nodes = outputnodes

        self.tanh = lambda x: np.tanh(x)
        self.relu = lambda x: np.maximum(x, 0)
        self.sigmoid = lambda x: scipy.special.expit(x)

        self.lr = learning_rate

        self.wie = np.random.normal(0.0, pow(self.encoding_nodes1, -0.5), (self.encoding_nodes1, self.input_nodes))
        self.wee = np.random.normal(0.0, pow(self.encoding_nodes2, -0.5), (self.encoding_nodes2, self.encoding_nodes1))
        self.wem = np.random.normal(0.0, pow(self.middle_nodes, -0.5), (self.middle_nodes, self.encoding_nodes2))
        self.wmd = np.random.normal(0.0, pow(self.decoding_nodes2, -0.5), (self.decoding_nodes2, self.middle_nodes))
        self.wdd = np.random.normal(0.0, pow(self.decoding_nodes1, -0.5), (self.decoding_nodes1, self.decoding_nodes2))
        self.wdo = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.decoding_nodes1))

    def train(self, input_list, target_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        encoding_layer1_input =  np.dot(self.wie, inputs)
        encoding_layer1_output = self.tanh(encoding_layer1_input)

        encoding_layer2_input = np.dot(self.wee, encoding_layer1_output)
        encoding_layer2_output = self.tanh(encoding_layer2_input)

        middle_layer_input = np.dot(self.wem, encoding_layer2_output)
        middle_layer_output = self.tanh(middle_layer_input)

        decoding_layer2_input = np.dot(self.wmd, middle_layer_output)
        decoding_layer2_output = self.tanh(decoding_layer2_input)

        decoding_layer1_input = np.dot(self.wdd, decoding_layer2_output)
        decoding_layer1_output = self.tanh(decoding_layer1_input)

        final_input = np.dot(self.wdo, decoding_layer1_output)
        final_output = self.tanh(final_input)

        #layer error
        output_error = targets - final_output
        decoding_layer1_error = np.dot(self.wdo.T, output_error)
        decoding_layer2_error = np.dot(self.wdd.T, decoding_layer1_error)
        middle_layer_error = np.dot(self.wmd.T, decoding_layer2_error)
        encoding_layer2_error = np.dot(self.wem.T, middle_layer_error)
        encoding_layer1_error = np.dot(self.wee.T, encoding_layer2_error)

        #weight update
        self.wdo += self.lr * np.dot(output_error * (1 - (np.tanh(final_output) ** 2)), np.transpose(decoding_layer1_output))
        self.wdd += self.lr * np.dot(decoding_layer1_error * (1 - (np.tanh(decoding_layer1_output) ** 2)) , np.transpose(decoding_layer2_output))
        self.wmd += self.lr * np.dot(decoding_layer2_error * (1 - (np.tanh(decoding_layer2_output) ** 2)), np.transpose(middle_layer_output))
        self.wem += self.lr * np.dot(middle_layer_error * (1 - (np.tanh(middle_layer_output) ** 2)), np.transpose(encoding_layer2_output))
        self.wee += self.lr * np.dot(encoding_layer2_error * (1 - (np.tanh(encoding_layer2_output) ** 2 )), np.transpose(encoding_layer1_output))
        self.wie += self.lr * np.dot(encoding_layer1_error * (1 - (np.tanh(encoding_layer1_output) ** 2)), np.transpose(inputs))

inputnodes = 21
el1 = 4
el2 = 2
ml = 1
dl2 = 2
dl1 = 4
outputnodes = 21

learning_rate = 0.01

nn = Net(inputnodes, el1, el2, ml, dl2, dl1, outputnodes, learning_rate)

#load the data
training_data_file = open("w_finger.in", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 61

for e in range(epochs):
    for record in training_data_list:
        all_records = record.split('  ')
        inputs = np.asfarray(all_records[0:])
        targets = inputs
        nn.train(inputs, targets)
    print(e)

pickle_wie = open('weights2/wie.pickle', 'wb')
pickle_wee = open('weights2/wee.pickle', 'wb')
pickle_wem = open('weights2/wem.pickle', 'wb')
pickle_wmd = open('weights2/wmd.pickle', 'wb')
pickle_wdd = open('weights2/wdd.pickle', 'wb')
pickle_wdo = open('weights2/wdo.pickle', 'wb')

pickle.dump(nn.wie, pickle_wie)
pickle.dump(nn.wee, pickle_wee)
pickle.dump(nn.wem, pickle_wem)
pickle.dump(nn.wmd, pickle_wmd)
pickle.dump(nn.wdd, pickle_wdd)
pickle.dump(nn.wdo, pickle_wdo)
