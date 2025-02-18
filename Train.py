# Script for Neural Network training
import os
import sys 
import cv2
import numpy as np
import neurolab as nl

num_data = 50
input_file_name = 'letter.data'
labels_dictionary = 'onamdig'
num_labels_dictionary = len(labels_dictionary)

num_train = int(0.9 * num_data)
num_test = num_data - num_train
input_data_start_index = 6
input_data_end_index = -1

input_data = []
labels = []

with open(input_file_name, 'r') as file:
    for line in file.readlines():
        list_vals = line.split('\t')
        if list_vals[1] not in labels_dictionary: 
            continue
        label = np.zeros((num_labels_dictionary,1))
        label[labels_dictionary.index(list_vals[1])] = 1
        labels.append(label)

        current_char = np.array([float(x) for x in list_vals[input_data_start_index:input_data_end_index]])
        input_data.append(current_char)

        if len(input_data) >= num_data: 
            break

input_data = np.asfarray(input_data)
labels = np.array(labels).reshape(num_data, num_labels_dictionary)

num_dims = len(input_data[0])

neural_network = nl.net.newff([[0,1] for _ in range(len(input_data[0]))], [128, 16, num_labels_dictionary])

neural_network.trainf = nl.train.train_gd

error_progress = neural_network.train(input_data[:num_train, :], labels[:num_train,:], epochs = 5000, show = 1000, goal= 0.01)

# Saving neural network to the 'neural_network.input_data' file
neural_network.save('neural_network.data')

pred_test = neural_network.sim(input_data[num_train:, :])

for i in range(num_test):
    print('\n Original: ', labels_dictionary[np.argmax(labels[num_train + i])])
    print('\n Predicted: ', labels_dictionary[np.argmax(pred_test[i])])
