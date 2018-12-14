import numpy as np
import pickle
import matplotlib.pyplot as plt

activation_function = lambda x: np.tanh(x)

pickle_wie = open('weights2/wie.pickle', 'rb')
pickle_wee = open('weights2/wee.pickle', 'rb')
pickle_wem = open('weights2/wem.pickle', 'rb')
pickle_wmd = open('weights2/wmd.pickle', 'rb')
pickle_wdd = open('weights2/wdd.pickle', 'rb')
pickle_wdo = open('weights2/wdo.pickle', 'rb')

wie = pickle.load(pickle_wie)
wee = pickle.load(pickle_wee)
wem = pickle.load(pickle_wem)
wmd = pickle.load(pickle_wmd)
wdd = pickle.load(pickle_wdd)
wdo = pickle.load(pickle_wdo)

def query(input_list):
        inputs = np.array(input_list, ndmin=2).T
        
        encoding_layer1_input =  np.dot(wie, inputs)
        encoding_layer1_output = np.tanh(encoding_layer1_input)

        encoding_layer2_input = np.dot(wee, encoding_layer1_output)
        encoding_layer2_output = np.tanh(encoding_layer2_input)

        middle_layer_input = np.dot(wem, encoding_layer2_output)
        middle_layer_output = np.tanh(middle_layer_input)

        decoding_layer2_input = np.dot(wmd, middle_layer_output)
        decoding_layer2_output = np.tanh(decoding_layer2_input)

        decoding_layer1_input = np.dot(wdd, decoding_layer2_output)
        decoding_layer1_output = np.tanh(decoding_layer1_input)

        final_input = np.dot(wdo, decoding_layer1_output)
        final_outputs = np.tanh(final_input)

        return (final_outputs - inputs) ** 2
        #print(final_outputs,inputs)

test_data_file = open('w_finger.in', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

for record in test_data_list:
    all_values = record.split('  ')
    inputs = np.asfarray(all_values)
    outputs = query(inputs)
    sum = np.sum(outputs)
    print(outputs)
