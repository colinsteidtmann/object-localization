import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))
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
IMAGE_SIZE = 231


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
    
    #l2 weight decay Regularizer
    l2_w_decay = tf.keras.regularizers.l2(l=1e-5)

    #FeatureMapConvNet
    conv_layer1 = tf.keras.layers.SeparableConv2D(filters=96, kernel_size=[11, 11], strides=[4, 4], padding="valid",data_format="channels_last", activation="relu", kernel_regularizer=l2_w_decay)(inputs)
    pool_layer1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding="valid", data_format="channels_last")(conv_layer1)
    
    conv_layer2 = tf.keras.layers.SeparableConv2D(filters=256, kernel_size=[5, 5], strides=[1, 1], padding="valid",data_format="channels_last", activation="relu", kernel_regularizer=l2_w_decay)(pool_layer1)
    pool_layer2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding="valid", data_format="channels_last")(conv_layer2)

    conv_layer3 = tf.keras.layers.SeparableConv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding="same", data_format="channels_last", activation="relu", kernel_regularizer=l2_w_decay)(pool_layer2)
    conv_layer4 = tf.keras.layers.SeparableConv2D(filters=1024, kernel_size=[3, 3], strides=[1, 1], padding="same", data_format="channels_last", activation="relu", kernel_regularizer=l2_w_decay)(conv_layer3)
    conv_layer5 = tf.keras.layers.SeparableConv2D(filters=1024, kernel_size=[3, 3], strides=[1, 1], padding="same", data_format="channels_last", activation="relu", kernel_regularizer=l2_w_decay)(conv_layer4)
    pool_layer5 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding="valid", data_format="channels_last")(conv_layer5)


    """boxConvnet branch"""
    box_conv_layer1 = tf.keras.layers.SeparableConv2D(filters=4096, kernel_size=[6, 6], strides=[1, 1], padding="valid",data_format="channels_last", activation="relu", kernel_regularizer=l2_w_decay)(pool_layer5)
    dropout_box_layer1 = tf.keras.layers.Dropout(rate=0.5)(box_conv_layer1)
    box_conv_layer2 = tf.keras.layers.SeparableConv2D(filters=1024, kernel_size=[1, 1], strides=[1, 1], padding="same", data_format="channels_last", activation="relu", kernel_regularizer=l2_w_decay)(dropout_box_layer1)
    dropout_box_layer2 = tf.keras.layers.Dropout(rate=0.5)(box_conv_layer2)
    box_conv_layer3 = tf.keras.layers.SeparableConv2D(filters=4, kernel_size=[1, 1], strides=[1, 1], padding="same", data_format="channels_last", activation="linear", kernel_regularizer=l2_w_decay)(dropout_box_layer2)
    box_conv_layer3 = tf.keras.layers.Flatten(name='box_output')(box_conv_layer3)

    """classConvnet branch"""
    class_conv_layer1 = tf.keras.layers.SeparableConv2D(filters=4096, kernel_size=[6, 6], strides=[1, 1], padding="valid", data_format="channels_last", activation="relu")(pool_layer5)
    dropout_class_layer1 = tf.keras.layers.Dropout(rate=0.5)(class_conv_layer1)
    class_conv_layer2 = tf.keras.layers.SeparableConv2D(filters=1024, kernel_size=[1, 1], strides=[1, 1], padding="same", data_format="channels_last", activation="relu")(dropout_class_layer1)
    dropout_class_layer2 = tf.keras.layers.Dropout(rate=0.5)(class_conv_layer2)
    class_conv_layer3 = tf.keras.layers.SeparableConv2D(filters=NUM_CLASSES, kernel_size=[1, 1], strides=[1, 1], padding="same", data_format="channels_last", activation="softmax")(dropout_class_layer2)
    class_conv_layer3 = tf.keras.layers.Flatten(name='class_output')(class_conv_layer3)

    model_optimizer = tf.keras.optimizers.Nadam(lr=0.001)
    model = tf.keras.Model(inputs=inputs, outputs=[box_conv_layer3, class_conv_layer3])
    model.compile(optimizer=model_optimizer,
                loss=[tf.keras.losses.MeanAbsoluteError(), tf.keras.losses.CategoricalCrossentropy()],
                loss_weights={'box_output': 1.,
                        'class_output': 1.},
                metrics={'box_output': 'mean_absolute_error',
                        'class_output': 'categorical_crossentropy'})


    return model


def main():
    model = create_model()
    train_datagen = DataGenerator(TRAIN_CSV)
    val_generator = DataGenerator(VALIDATION_CSV)
    train_images, train_boxes, train_classes = train_datagen.getitems()
    test_images, test_boxes, test_classes = val_generator.getitems()

    """ Labels one hot encoded """
    train_labels_one_hot = tf.keras.utils.to_categorical(train_classes, num_classes=NUM_CLASSES)
    test_labels_one_hot = tf.keras.utils.to_categorical(test_classes, num_classes=NUM_CLASSES)

    """ training """
    n_epochs = 100
    batch_size = 32

    model.fit(x=train_images, y=[train_boxes,train_classes], batch_size=batch_size, epochs=n_epochs, shuffle=True, verbose=2, validation_data=(test_images,[test_boxes,test_classes]))
    
    """ testing """
    score = model.evaluate(test_images, [test_boxes, test_classes], verbose=0)
    print(score)

    test_image_predictions = model.predict(test_images[22:23])
    predicted_box_coords, predicted_class = np.squeeze(test_image_predictions[0]), np.argmax(test_image_predictions[1])
    
    img = Image.fromarray(test_images[22].astype(np.uint8()), 'RGB')
    draw = ImageDraw.Draw(img)
    draw.rectangle(predicted_box_coords.tolist())
    img.save('predicted.png')
    img.show()

    img = Image.fromarray(test_images[22].astype(np.uint8()), 'RGB')
    draw = ImageDraw.Draw(img)
    draw.rectangle(np.array(test_boxes[22]).tolist())
    img.save('actual.png')
    img.show()

    print("predicted_class = ", predicted_class, "actual class = ", test_classes[22])
        

if __name__ == "__main__":
    main()
