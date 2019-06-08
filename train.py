import csv
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from NeuralNets import ConvNet as cnn
from NeuralNets import NeuralNetwork as neuralnetwork
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
    fmapConvnet = cnn.ConvNet(
        conv_method="convolution",
        layer_names=["conv", "pool", "conv", "pool",
                     "conv", "pool", "conv", "pool", "conv", "pool"],
        num_filters=[8, None, 8, None, 6, None, 6, None, 6, None],
        kernel_sizes=[[5, 5], None, [5, 5], None, [
            4, 4], None, [4, 4], None, [4, 4], None],
        stride_sizes=[[1, 1], [2, 2], [1, 1], [2, 2], [
            1, 1], [2, 2], [1, 1], [2, 2], [1, 1], [2, 2]],
        pool_sizes=[None, [2, 2], None, [2, 2], None,
                    [2, 2], None, [2, 2], None, [2, 2]],
        pool_fns=[None, "max", None, "max", None,
                  "max", None, "max", None, "max"],
        pad_fns=["valid", "valid", "valid", "valid", "valid",
                 "valid", "valid", "valid", "valid", "valid"],
        activations=["relu", None, "relu", None,
                     "relu", None, "relu", None, "relu", None],
        input_channels=3,
        scale_method=None,
        optimizer="nadam",
        lr=0.005,
        lr_decay=(0.0)
    )
    boxConvnet = cnn.ConvNet(
        conv_method="convolution",
        layer_names=["conv", "conv", "conv"],
        num_filters=[256, 128, 4],
        kernel_sizes=[[4, 4], [1, 1], [1, 1]],
        stride_sizes=[[1, 1], [1, 1], [1, 1]],
        pool_sizes=[None, None, None],
        pool_fns=[None, None, None],
        pad_fns=["valid", "same", "same"],
        activations=["relu", "relu", "linear"],
        input_channels=6,
        scale_method="normalize",
        optimizer="nadam",
        lr=0.005,
        lr_decay=(0.0)
    )
    classConvnet = cnn.ConvNet(
        conv_method="convolution",
        layer_names=["conv", "conv", "conv"],
        num_filters=[256, 128, NUM_CLASSES],
        kernel_sizes=[[4, 4], [1, 1], [1, 1]],
        stride_sizes=[[1, 1], [1, 1], [1, 1]],
        pool_sizes=[None, None, None],
        pool_fns=[None, None, None],
        pad_fns=["valid", "same", "same"],
        activations=["relu", "relu", "softmax"],
        input_channels=6,
        scale_method="normalize",
        optimizer="nadam",
        lr=0.005,
        lr_decay=(0.0)
    )
    boxNN = neuralnetwork.NeuralNetwork(
        [fmapConvnet, boxConvnet], loss_fn="mean_squared_error")
    classNN = neuralnetwork.NeuralNetwork(
        [fmapConvnet, classConvnet], loss_fn="cross_entropy")

    return boxNN, classNN


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
            X_batch, true_boxes_batch, true_classes_batch = X[
                batch_idx], true_boxes[batch_idx], true_classes[batch_idx]
            yield X_batch, true_boxes_batch, true_classes_batch

    n_epochs = 10
    batch_size = 100
    for epoch in range(n_epochs):
        for X_batch, true_boxes_batch, true_classes_batch in shuffle_batch(train_images, train_boxes, train_labels_one_hot, batch_size):
            boxNN.check_gradients(X_batch, true_boxes_batch)
            classNN.check_gradients(X_batch, true_classes_batch)
            boxNN.sgd_fit(X_batch, true_boxes_batch,
                          batch_size=batch_size, shuffle_inputs=False)
            classNN.sgd_fit(X_batch, true_classes_batch,
                            batch_size=batch_size, shuffle_inputs=False)

        train_box_predictions = boxNN.feedforward(
            train_images, scale=True, test=True)
        train_classes_predictions = classNN.feedforward(
            train_images, scale=True, test=True)

        test_box_predictions = boxNN.feedforward(
            test_images, scale=True, test=True)
        test_classes_predictions = classNN.feedforward(
            test_images, scale=True, test=True)

        train_box_loss = boxNN.get_losses(train_box_predictions, train_boxes)
        test_box_loss = boxNN.get_losses(test_box_predictions, test_boxes)

        train_class_pct_correct = np.mean(np.squeeze(
            np.argmax(train_predictions, 1)) == np.argmax(train_labels_one_hot, 1))
        test_class_pct_correct = np.mean(np.squeeze(
            np.argmax(test_predictions, 1)) == np.argmax(test_labels_one_hot, 1))
        print(epoch, "Last batch class accuracy:", train_class_pct_correct, "Test class accuracy:",
              test_class_pct_correct, "Last batch boxes loss:", train_box_loss, "Test boxes loss:", test_box_loss)


if __name__ == "__main__":
    main()
