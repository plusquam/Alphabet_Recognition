# Script for Neural Network testing
import os
import sys 
import cv2
import numpy as np
import neurolab as nl

##################### Defines #######################################
display_pre_images = False
display_test_results = True
# Set to True to read test data from input_file_name file. Set to False to read test images from test_images_directory directory
read_from_file = False

test_images_directory = 'test_data'
converted_images_directory = 'test_data\\converted_images\\'

num_data = 10000
input_file_name = 'letter.data'
output_log_file_name = 'test_results.txt'
labels_dictionary = 'onamdig'
num_labels_dictionary = len(labels_dictionary)
num_data_rows = 16
num_data_cols = 8

num_train = int(0.9 * num_data)
num_test = num_data - num_train
input_data_start_index = 6
input_data_end_index = -1

##################### Functions #######################################

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

def read_data_from_file():
    return_data = []
    return_labels = []
    invalid_data = 0
    
    with open(input_file_name, 'r') as file:
        for line in file.readlines():
            list_vals = line.split('\t')
            if list_vals[1] not in labels_dictionary:
                invalid_data += 1
                continue
            label = np.zeros((num_labels_dictionary,1))
            label[labels_dictionary.index(list_vals[1])] = 1
            return_labels.append(label)

            current_char = np.array([int(x) for x in list_vals[input_data_start_index:input_data_end_index]])
            return_data.append(current_char)

            if len(return_data) >= num_data: 
                break

    return_data = np.asfarray(return_data)
    return_labels = np.array(return_labels).reshape(num_data, num_labels_dictionary)

    return return_data, return_labels, invalid_data


def read_and_convert_images():
    # Reading images
    input_images_name_list = get_files_from_directory(test_images_directory)
    num_input_images = len(input_images_name_list)

    output_images_data_list = []
    labels_list = []
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
            if display_pre_images:
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
            if display_pre_images:
                cv2.namedWindow('tresholded_image', cv2.WINDOW_NORMAL)
                cv2.imshow('tresholded_image', tresholded_image)

            #First stage of resizing
            if input_image.shape[0] > num_data_rows*10:
                resized_image = cv2.resize(tresholded_image, dsize=(num_data_cols*10, num_data_rows*10), interpolation=cv2.INTER_CUBIC)
                #ret, resized_image = cv2.threshold(resized_image, 210, 255, cv2.THRESH_BINARY)
                if display_pre_images:
                    cv2.namedWindow('first_stage_resized_image', cv2.WINDOW_NORMAL)
                    cv2.imshow('first_stage_resized_image', resized_image)
                print("Image " + str(index) + " first stage resize size: " + str(resized_image.shape))
            else:
                resized_image = tresholded_image

            # Resizing image to 16x8 size
            resized_image = cv2.resize(resized_image, dsize=(num_data_cols, num_data_rows), interpolation=cv2.INTER_CUBIC)
            ret, resized_image = cv2.threshold(resized_image, 180, 255, cv2.THRESH_BINARY)
            if display_pre_images:
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
            if display_pre_images:
                cv2.waitKey(1000)
            index += 1

    output_images_data_list = np.array(output_images_data_list)
    labels_list = np.array(labels_list)

    cv2.destroyAllWindows()
    return output_images_data_list, labels_list, valid_filenames_list, (num_input_images - len(valid_filenames_list))


def save_test_log(filename, total, passed, unreaded):
    with open(filename, 'w') as file:
        file.write('Test results:\n')
        file.write('Total:    ' + str(total) + '\n')
        file.write('Passed:   ' + str(passed) + " - " + str(passed * 100.0 / (total - unreaded)) + '%\n')
        file.write('Failed:   ' + str(total - passed - unreaded) + '\n')
        file.write('Invalid:  ' + str(unreaded) + '\n')

        print('\nTest results:\n')
        print('Total:    ' + str(total))
        print('Passed:   ' + str(passed) + " - " + str(passed * 100.0 / (total - unreaded)) + '%')
        print('Failed:   ' + str(total - passed - unreaded))
        print('Invalid:  ' + str(unreaded))
        file.close()


##################### Main #######################################

input_data = []
labels = []
filenames = []

# Test statistic variables
test_num_passed = 0
test_num_total  = 0
test_num_unreaded = 0

if read_from_file:
    input_data, labels, test_num_unreaded = read_data_from_file()
    test_num_total = input_data.shape[0] + test_num_unreaded
else:
    input_data, labels, filenames, test_num_unreaded = read_and_convert_images()
    test_num_total = len(filenames) + test_num_unreaded

# Loading Neural Network
neural_network = nl.load('neural_network.data')

# Letters detection 
pred_test = neural_network.sim(input_data)

# Results
for i in range(len(labels)):
    labels_index = np.argmax(labels[i])
    prediction_index = np.argmax(pred_test[i])

    # Checking whether test was passed
    if_passed = labels_index == prediction_index
    if if_passed:
        test_num_passed += 1

    if read_from_file:
        print('\nImage: ' + str(i))
    else:
        print('\nImage: ' + filenames[i])

    print('Original: ', labels_dictionary[labels_index])
    print('Predicted: ', labels_dictionary[prediction_index])

    # Displaying results on images
    if display_test_results and not read_from_file:
        # Reading original image
        image = cv2.imread(test_images_directory + '\\' + filenames[i], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize=(400, 800), interpolation=cv2.INTER_CUBIC)
        ret, image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Overlaying postprocessed image
        post_image = cv2.imread(converted_images_directory + filenames[i], cv2.IMREAD_GRAYSCALE)
        post_image = cv2.resize(post_image, dsize=(60, 120), interpolation=cv2.INTER_CUBIC)
        ret, post_image = cv2.threshold(post_image, 120, 255, cv2.THRESH_BINARY)
        post_image = cv2.cvtColor(post_image, cv2.COLOR_GRAY2RGB)

        post_image_offset_x = image.shape[1] - 80
        post_image_offset_y = image.shape[0] - 140
        image[post_image_offset_y:post_image_offset_y+post_image.shape[0], post_image_offset_x:post_image_offset_x+post_image.shape[1]] = post_image
        cv2.rectangle(image, (post_image_offset_x - 2, post_image_offset_y - 2), (post_image_offset_x+post_image.shape[1] + 2, post_image_offset_y+post_image.shape[0] + 2), (0,0,255), 2)

        # Displaying output results
        cv2.putText(image, filenames[i], (10,40), cv2.FONT_HERSHEY_DUPLEX , 0.8, (0,0,255), 2)

        if if_passed:
            cv2.putText(image, "Result: ", (150,40), cv2.FONT_HERSHEY_DUPLEX , 0.8, (0,255,0), 2)
            cv2.putText(image, str(labels_dictionary[prediction_index]), (240,40), cv2.FONT_HERSHEY_DUPLEX , 1.5, (0,255,0), 3)
        else:
            cv2.putText(image, "Result: ", (150,40), cv2.FONT_HERSHEY_DUPLEX , 0.8, (0,0,255), 2)
            cv2.putText(image, str(labels_dictionary[prediction_index]), (240,40), cv2.FONT_HERSHEY_DUPLEX , 1.5, (0,0,255), 3)
    
        cv2.namedWindow('test_results', cv2.WINDOW_NORMAL)
        cv2.imshow('test_results',image)
        cv2.waitKey(2000)
 
# Saving logs from tests
save_test_log(output_log_file_name, test_num_total, test_num_passed, test_num_unreaded)
cv2.destroyAllWindows()

