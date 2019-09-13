# Visualization
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import cv2
import numpy as np
import os
import Model

from sklearn.model_selection import train_test_split


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    print(int(true_label), predicted_label, predictions_array, 100 * np.max(predictions_array))
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[int(true_label)]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(2), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[int(true_label)].set_color('blue')


def locate():
    data = cv2.cvtColor(cv2.imread("1.jpg"), cv2.COLOR_BGR2RGB)

    heatmap = heatmodel.predict(data.reshape(1, data.shape[0], data.shape[1], data.shape[2]))

    plt.imshow(heatmap[0, :, :, 0])
    plt.title("Heatmap")
    plt.show()
    plt.imshow(heatmap[0, :, :, 0] > 0.99, cmap="gray")
    plt.title("Car Area")
    plt.show()

    xx, yy = np.meshgrid(np.arange(heatmap.shape[2]), np.arange(heatmap.shape[1]))
    x = (xx[heatmap[0, :, :, 0] > 0.99])
    y = (yy[heatmap[0, :, :, 0] > 0.99])

    for i, j in zip(x, y):
        cv2.rectangle(data, (i * 8, j * 8), (i * 8 + 64, j * 8 + 64), (0, 0, 255), 5)
    return data


X, Y = [], []

plt.imshow(X[1])

X = np.array(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
input_shape = (3, 64, 64)

model = Model.get_conv()

model.fit(X_train, Y_train, batch_size=32, epochs=15, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
model.save_weights("localize4.h5")

class_names = ['Not Waldo', 'Waldo']
predictions = model.predict(X_test)

# for x in range(100):
#     print(predictions[x], Y_test[x])
#     plt.title("Augmented")
#     plt.imshow(X_test[x])
#     plt.show()

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 10
num_cols = 3
num_images = num_rows * num_cols * 2
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, Y_test, X_test)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, Y_test)

# Grab an image from the test dataset
img = X_test[0]

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))

predictions_single = model.predict(img)
plot_value_array(0, predictions_single, Y_test)
_ = plt.xticks(range(10), class_names, rotation=45)

heatmodel = Model.get_conv(input_shape=(None, None, 3), filename="localize.h5")

annotated = locate()

plt.title("Augmented")
plt.imshow(annotated)
plt.show()
