# Visualization
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import numpy as np
import random
import os
import time

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D

input_shape = (3, 64, 64)


def get_conv(input_shape=(64, 64, 3), filename=None):
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape, output_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', name='conv1', input_shape=input_shape, padding="same"))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv2', padding="same"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    # model.add(Conv2D(64, (8, 8), activation='relu', name='conv3', padding="same"))
    # model.add(MaxPooling2D(pool_size=(4,4)))
    model.add(Conv2D(128, (8, 8), activation="relu", name="dense1"))  # This was Dense(128)
    model.add(Dropout(0.5))
    model.add(Conv2D(1, (14, 14), name="dense2", activation="sigmoid"))  # This was Dense(1)

    # for layer in model.layers:
    #     print(layer.input_shape, layer.output_shape)
    if filename:
        model.load_weights(filename)
    return model


model = get_conv()
model.add(Flatten())
model.compile(loss='mse', optimizer='adadelta', metrics=['accuracy'])

heatmodel = get_conv(input_shape=(None, None, 3), filename="Models/localize7.h5")


def store_image(path, name, x, y):
    image = cv2.imread(path)[x:x + 64, y:y + 64]
    # while True:
    # cv2.imshow('cutout', image)
    # k = cv2.waitKey(33)
    # if k == 32:  # space to mark false
    #     image_PIL = Image.open(path).crop((x, y, x + 64, y + 64))
    #     image_PIL.save('Data/Predictions/NotWaldos/' + name + str(x) + '-' + str(y) + '.png')
    #     break
    # elif k == -1:  # normally -1 returned,so don't print it
    #     continue
    # elif k == 13:
    #     image_PIL = Image.open(path).crop((x, y, x + 64, y + 64))
    #     image_PIL.save('Data/Predictions/NotWaldos/' + name + str(x) + '-' + str(y) + '.png')

    image_PIL = Image.open(path).crop((x, y, x + 64, y + 64))
    image_PIL.save('Data/Predictions/' + name + str(x) + '-' + str(y) + '.png')


def locate(img):
    num_sub_img = 100
    data = cv2.cvtColor(cv2.imread("Data/Raw/Batch1/" + img), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(cv2.imread("Data/Raw/Batch1/" + img), cv2.COLOR_BGR2GRAY)
    # Image.fromarray(gray).show()
    coloured_heat = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # plt.imshow(gray)
    # plt.show()

    heatmap = heatmodel.predict(data.reshape(1, data.shape[0], data.shape[1], data.shape[2]))
    height, width = data.shape[:2]
    pred_height, pred_width = heatmap.shape[1:3]
    plt.imshow(heatmap[0, :, :, 0])
    plt.title("Heatmap")
    plt.show()
    # plt.imshow(heatmap[0, :, :, 0] > 0.95, cmap="gray")
    # plt.title("Car Area")
    # plt.show()
    # coloured_heat = color_gray(data, heatmap, gray).astype(np.uint8)
    # Image.fromarray(coloured_heat).show()

    xx, yy = np.meshgrid(np.arange(heatmap.shape[2]), np.arange(heatmap.shape[1]))
    x = (xx[heatmap[0, :, :, 0] > 0.99])
    y = (yy[heatmap[0, :, :, 0] > 0.99])

    # print('x', x)
    chosen_sub_images = random.sample([[x[num], y[num]] for num in range(len(x))], min(num_sub_img, len(x)))

    plt.title("Augmented")
    plt.imshow(data)
    plt.show()

    # for i, j in chosen_sub_images:
    #     store_image('Data/Raw/Batch1/' + img, img, int(round(i * height / pred_height, 0)),
    #                 int(round(j * width / pred_width, 0)))

    now = time.time()
    for i, j in zip(x, y):
        y_pos=j*3
        x_pos=i*3
        # y_pos = int(j * height / pred_height)
        # y_pos_max=int((j+1) * height / pred_height)-1
        # x_pos = int(i * width / pred_width)
        # x_pos_max = int((i+1) * width / pred_width)-1
        # print('y', y_pos, height, pred_height, j)
        # print('x', x_pos, width, pred_width, i)
        # coloured_heat = color_gray_pos(data, 1, gray, coloured_heat, x_pos, x_pos_max,
        #                                y_pos, y_pos_max)
        cv2.rectangle(data, (x_pos, y_pos),
                                    (x_pos+64, y_pos + 64), (0, 0, 255), 5)
        # cv2.rectangle(data, (int(i * height / pred_height), int(j * width / pred_width)),
        #               (int(i * height / pred_height)+64, int(j * width / pred_width) + 64), (0, 0, 255), 5)
        # cv2.rectangle(data, (i * height // pred_height, j * width // pred_width),
        #               (i * height // pred_height + 64, j * width // pred_width + 64), (0, 0, 255), 5)

    coloured_heat = coloured_heat.astype(np.uint8)
    print(time.time() - now)
    # Image.fromarray(coloured_heat).show()
    # if random.randint(0, 10) < 1:
    #     store_image('Data/Raw/Batch1/'+img, img, i, j)
    return data


def color_gray_pos(image, percent, gray, coloured_heat, start_x, end_x, start_y, end_y):
    for x in range(start_x, end_x):
        for y in range(start_y, end_y):
            for c in range(3):
                coloured_heat[y, x, c] = np.uint8(int(gray[y, x] + percent * (int(image[y, x, c] - gray[y, x]))))
    return coloured_heat
    # return np.array([[[int(
    #     int(gray[y, x]) + percent * (
    #             int(image[y, x, c]) - gray[y, x])) for c in range(3)] for x in range(start_x, end_x)] for y in
    #     range(start_y, end_y)])


def color_gray(image, heatmap, gray):
    height, width = image.shape[:2]
    heat_height, heat_width = heatmap.shape[1:3]
    print(height, width, heat_height, heat_width)
    return np.array([[[int(
        int(gray[y, x]) + heatmap[0, int(y * heat_height / height), int(x * heat_width / width), 0] * (
                int(image[y, x, c]) - gray[y, x])) for c in range(3)] for x in range(width)] for y in
        range(height)])


# for img in os.listdir("Data/Raw/Test/"):
#     print(img)
#     try:
#         annotated = locate(img)
#         Image.fromarray(annotated).show()
#         plt.title("Augmented")
#         plt.imshow(annotated)
#         plt.show()
#     except Exception as e:
#         print('exception', e)
annotated = locate('14.jpg')
Image.fromarray(annotated).show()
print(annotated.shape[:2])
plt.title("Augmented")
plt.imshow(annotated)
plt.show()
