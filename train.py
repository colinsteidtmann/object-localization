import tensorflow as tf
import csv
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import random

TRAIN_CSV = "train.csv"
VALIDATION_CSV = "validation.csv"
NUM_CLASSES = 37
IMAGE_SIZE = 224


class DataGenerator():
    def __init__(self, csv_file, shuffle=True):
        self.boxes = []
        self.classes = []

        with open(csv_file, "r") as file:
            reader = csv.reader(file, delimiter=",")
            for index, row in enumerate(reader):
                for i, r in enumerate(row[1:7]):
                    row[i+1] = int(r)

                path, image_height, image_width, x0, y0, x1, y1, _, _ = row
                self.boxes.append((path, x0, y0, x1, y1))
                self.classes.append(int(row[-1]))

        random_idx = np.arange(0, len(self.boxes))
        if shuffle:
            random.shuffle(random_idx)
        self.boxes = np.array(self.boxes)[random_idx]
        self.classes = np.array(self.classes)[random_idx]

    def getitems(self):
        boxes = self.boxes
        classes = self.classes

        batch_images = np.zeros(
            (len(boxes), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        batch_boxes = np.zeros((len(boxes), 4), dtype=np.float32)
        batch_classes = np.zeros((len(boxes)))
        for i, (row, class_idx) in enumerate(zip(boxes, classes)):
            path, x0, y0, x1, y1 = row
            with Image.open(path) as img:
                image_width = img.width
                image_height = img.height

                x_scale = IMAGE_SIZE / image_width
                y_scale = IMAGE_SIZE / image_height
                x0 = int(np.round(int(x0)*x_scale))
                y0 = int(np.round(int(y0)*y_scale))
                x1 = int(np.round(int(x1)*x_scale))
                y1 = int(np.round(int(y1)*y_scale))

                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                img = img.convert('RGB')
                img = np.array(img, dtype=np.float32)

            batch_images[i] = img
            batch_boxes[i] = x0, y0, x1, y1
            batch_classes[i] = int(class_idx)

        return batch_images, batch_boxes, batch_classes


def create_model():
    #Inputs
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    
    #FeatureMapConvNet
    conv_layer1 = tf.keras.layers.Conv2D(filters=8, kernel_size=[5, 5], strides=[1, 1], padding="valid",data_format="channels_last", activation="relu")(inputs)
    pool_layer1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding="valid", data_format="channels_last")(conv_layer1)
    
    conv_layer2 = tf.keras.layers.Conv2D(filters=8, kernel_size=[5, 5], strides=[1, 1], padding="valid",data_format="channels_last", activation="relu")(pool_layer1)
    pool_layer2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding="valid", data_format="channels_last")(conv_layer2)
    
    conv_layer3 = tf.keras.layers.Conv2D(filters=6, kernel_size=[4, 4], strides=[1, 1], padding="valid",data_format="channels_last", activation="relu")(pool_layer2)
    pool_layer3 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding="valid", data_format="channels_last")(conv_layer3)
    
    conv_layer4 = tf.keras.layers.Conv2D(filters=6, kernel_size=[4, 4], strides=[1, 1], padding="valid",data_format="channels_last", activation="relu")(pool_layer3)
    pool_layer4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding="valid", data_format="channels_last")(conv_layer4)
    
    conv_layer5 = tf.keras.layers.Conv2D(filters=6, kernel_size=[4, 4], strides=[1, 1], padding="valid",data_format="channels_last", activation="relu")(pool_layer4)
    pool_layer5 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding="valid", data_format="channels_last")(conv_layer5)
    norm_pool_layer5 = tf.keras.layers.BatchNormalization(axis=-1)(pool_layer5)
    
    """boxConvnet branch"""
    box_conv_layer1 = tf.keras.layers.Conv2D(filters=256, kernel_size=[4, 4], strides=[1, 1], padding="valid",data_format="channels_last", activation="relu")(norm_pool_layer5)
    box_conv_layer2 = tf.keras.layers.Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding="same",data_format="channels_last", activation="relu")(box_conv_layer1)
    box_conv_layer3 = tf.keras.layers.Conv2D(filters=4, kernel_size=[1, 1], strides=[1, 1], padding="same", data_format="channels_last", activation="linear")(box_conv_layer2)
    
    """classConvnet branch"""
    class_conv_layer1 = tf.keras.layers.Conv2D(filters=256, kernel_size=[4, 4], strides=[1, 1], padding="valid",data_format="channels_last", activation="relu")(norm_pool_layer5)
    class_conv_layer2 = tf.keras.layers.Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding="same",data_format="channels_last", activation="relu")(class_conv_layer1)
    class_conv_layer3 = tf.keras.layers.Conv2D(filters=NUM_CLASSES, kernel_size=[1, 1], strides=[1, 1], padding="same", data_format="channels_last", activation="softmax")(class_conv_layer2)


    box_model = tf.keras.Model(inputs=inputs, outputs=box_conv_layer3)
    box_model.compile('sgd', loss=tf.keras.losses.MeanSquaredError())
    class_model = tf.keras.Model(inputs=inputs, outputs=class_conv_layer3)
    class_model.compile('sgd', loss=tf.keras.losses.CategoricalCrossentropy())

    return box_model, class_model


def main():
    boxNN, classNN = create_model()
    train_datagen = DataGenerator(TRAIN_CSV)
    val_generator = DataGenerator(VALIDATION_CSV)
    train_images, train_boxes, train_classes = train_datagen.getitems()
    test_images, test_boxes, test_classes = val_generator.getitems()

    """ Labels one hot encoded """
    train_labels_one_hot = np.zeros((len(train_classes), NUM_CLASSES))
    for idx, hot_idx in enumerate(train_classes):
        train_labels_one_hot[idx, int(hot_idx)] = 1

    test_labels_one_hot = np.zeros((len(test_classes), NUM_CLASSES))
    for idx, hot_idx in enumerate(test_classes):
        test_labels_one_hot[idx, int(hot_idx)] = 1

    """ Training and testing """
    def shuffle_batch(X, true_boxes, true_classes, batch_size):
        rnd_idx = np.random.permutation(len(X))
        n_batches = len(X) // batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch, true_boxes_batch, true_classes_batch = X[batch_idx], true_boxes[batch_idx], true_classes[batch_idx]
            yield X_batch, true_boxes_batch, true_classes_batch

    

    n_epochs = 10
    batch_size = 100
    boxNN.fit(train_images, train_boxes, batch_size=batch_size, epochs=epochs, shuffle=True)
    classNN.fit(train_images, train_labels_one_hot, batch_size=batch_size, epochs=epochs, shuffle=True)
    
    score = classNN.evaluate(test_images, test_labels_one_hot, verbose=0)
    print('Test class loss:', score[0])
    print('Test class accuracy:', score[1])
        

if __name__ == "__main__":
    main()
