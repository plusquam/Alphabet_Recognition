# Script for Neural Network testing
import os
import sys 
import cv2
import numpy as np
import neurolab as nl

num_data = 50
input_f = 'letter.data'
orig_labels = 'onandig'
num_orig_labels = len(orig_labels)

num_train = int(0.9 * num_data)
num_test = num_data - num_train
start = 6
end = -1

data = []
labels = []

with open(input_f, 'r') as f:
    for line in f.readlines():
        list_vals = line.split('\t')
        if list_vals[1] not in orig_labels: 
            continue
        label = np.zeros((num_orig_labels,1))
        label[orig_labels.index(list_vals[1])] = 1
        labels.append(label)

        cur_char = np.array([float(x) for x in list_vals[start:end]])
        data.append(cur_char)

        if len(data) >= num_data: 
            break

data = np.asfarray(data)
labels = np.array(labels).reshape(num_data, num_orig_labels)

nn = nl.load('neural_network.data')

pred_test = nn.sim(data[num_train:, :])

for i in range(num_test):
    print('\n Original: ', orig_labels[np.argmax(labels[num_train + i])])
    print('\n Predicted: ', orig_labels[np.argmax(pred_test[i])])
