# Script for Neural Network testing
import os
import sys 
import cv2
import numpy as np
import neurolab as nl

test_images_directory = 'test_data'
converted_images_directory = 'test_data\\converted_images\\'

num_data = 50
input_file_name = 'letter.data'
labels_dictionary = 'onamdig'
num_labels_dictionary = len(labels_dictionary)
num_data_rows = 16
num_data_cols = 8

num_train = int(0.9 * num_data)
num_test = num_data - num_train
input_data_start_index = 6
input_data_end_index = -1

def get_files_from_directory(directory_name):
    file_list = []
    print("List of input files:")
    for file in os.listdir(directory_name):
        if file.endswith(".jpg"):
            file_list.append(file)
            print(os.path.join(directory_name, file))

    print("")
    return file_list


def to_square_image(image):
    width = image.shape[1]
    height = image.shape[0]
    crop_length = 0

    if width > height:
        crop_length = width - height
        if crop_length % 2 == 0:
            image = image[int(crop_length/2) : (width - int(crop_length/2))][:]
        else:
            image = image[int(crop_length/2) : (width - int(crop_length/2) - 1)][:]
    else:
        crop_length = height - width
        if crop_length % 2 == 0:
            image = image[:][int(crop_length/2) : (height - int(crop_length/2))]
        else:
            image = image[:][int(crop_length/2) : (height - int(crop_length/2) - 1)]

    return image

def get_label_from_filename(filename):
    return_status = False
    letters = filename.split('_')
    letter = letters[0]

    label = np.zeros((num_labels_dictionary), dtype=np.uint8)
    if letter in labels_dictionary: 
        label[labels_dictionary.index(letter)] = 1
        return_status = True
    else:
        return_status = False

    return return_status, label


def read_and_convert_images():
    # Reading images
    input_images_name_list = get_files_from_directory(test_images_directory)
    num_input_images = len(input_images_name_list)

    output_images_data_list = []#np.zeros((num_input_images, num_data_rows * num_data_rows), dtype=np.uint8)
    labels_list = []#np.zeros((num_input_images, num_data_rows * num_data_rows), dtype=np.uint8)
    valid_filenames_list = []
    

    index = 0
    for image_name in input_images_name_list:
        # Getting label from file name
        label_check, label = get_label_from_filename(image_name)

        # Checkin whether label was found in dictionary
        if label_check:
            valid_filenames_list.append(image_name)
            labels_list.append(np.array(label))

            # Reading image from file
            input_image = cv2.imread(test_images_directory + '\\' + image_name, cv2.IMREAD_GRAYSCALE)
            cv2.namedWindow('input_gray_image', cv2.WINDOW_NORMAL)
            cv2.imshow('input_gray_image',input_image)
            print("Image " + str(index) + " original size: " + str(input_image.shape))

            # Converting to square images
            #print("Image " + str(index) + " size before crop: [" + str(input_image.shape[1]) + "," + str(input_image.shape[0]) + "]")
            #square_image = to_square_image(input_image)
            #print("Image " + str(index) + " size after crop: [" + str(square_image.shape[1]) + "," + str(square_image.shape[0]) + "]")
            #cv2.imshow('input_gray_square_image',square_image)

            # Tresholding square image
            ret, tresholded_image = cv2.threshold(input_image, 120, 255, cv2.THRESH_BINARY)
            cv2.namedWindow('tresholded_image', cv2.WINDOW_NORMAL)
            cv2.imshow('tresholded_image', tresholded_image)

            #First stage of resizing
            resized_image = cv2.resize(tresholded_image, dsize=(num_data_cols*10, num_data_rows*10), interpolation=cv2.INTER_CUBIC)
            #ret, resized_image = cv2.threshold(resized_image, 210, 255, cv2.THRESH_BINARY)
            #cv2.namedWindow('first_stage_resized_image', cv2.WINDOW_NORMAL)
            cv2.imshow('first_stage_resized_image', resized_image)
            print("Image " + str(index) + " first stage resize size: " + str(resized_image.shape))

            # Resizing image to 16x8 size
            resized_image = cv2.resize(resized_image, dsize=(num_data_cols, num_data_rows), interpolation=cv2.INTER_CUBIC)
            ret, resized_image = cv2.threshold(resized_image, 180, 255, cv2.THRESH_BINARY)
            cv2.namedWindow('16x8_image', cv2.WINDOW_NORMAL)
            cv2.imshow('16x8_image', resized_image)
            print("Image " + str(index) + " final size: " + str(resized_image.shape))
            print("Image " + str(index) + " converted.")

            # Saving preprocessed image
            cv2.imwrite(converted_images_directory + image_name, resized_image)
            print("Image " + str(index) + " saved.")

            # Converting to Neural Network input data type
            ret, output_image = cv2.threshold(resized_image, 10, 1, cv2.THRESH_BINARY_INV)
            output_data = output_image.flatten()
            #output_data = np.array(output_image.reshape(1, num_data_rows * num_data_cols))

            output_images_data_list.append(output_data)

            print("")
            cv2.waitKey(1000)
            index += 1

    output_images_data_list = np.array(output_images_data_list)
    labels_list = np.array(labels_list)

    cv2.destroyAllWindows()
    return output_images_data_list, labels_list, valid_filenames_list


input_data = []
labels = []
filenames = []

input_data, labels, filenames = read_and_convert_images()

#with open(input_file_name, 'r') as file:
#    for line in file.readlines():
#        list_vals = line.split('\t')
#        if list_vals[1] not in labels_dictionary: 
#            continue
#        label = np.zeros((num_labels_dictionary,1))
#        label[labels_dictionary.index(list_vals[1])] = 1
#        labels.append(label)

#        current_char = np.array([int(x) for x in list_vals[input_data_start_index:input_data_end_index]])
#        input_data.append(current_char)

#        if len(input_data) >= num_data: 
#            break

#input_data = np.array(input_data)
#labels = np.array(labels).reshape(num_data, num_labels_dictionary)

neural_network = nl.load('neural_network.data')

pred_test = neural_network.sim(input_data)

for i in range(len(labels)):
    print('\nImage: ' + str(filenames[i]))
    print('Original: ', labels_dictionary[np.argmax(labels[i])])
    print('Predicted: ', labels_dictionary[np.argmax(pred_test[i])])
