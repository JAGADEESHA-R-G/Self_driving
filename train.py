import data_extraction
import numpy as np
import cv2
import imageio
import scipy
import os
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.utils import Sequence
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import math


def get_model():
    input_tensor = tf.keras.Input(shape=(66, 200, 3))

    layer_1 = layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation="relu",padding="valid",
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                            bias_initializer=keras.initializers.Constant(0.1), kernel_regularizer=l2(0.001))(input_tensor)
    layer_2 = layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation="relu",padding="valid",
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                            bias_initializer=keras.initializers.Constant(0.1), kernel_regularizer=l2(0.001))(layer_1)
    layer_3 = layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation="relu",padding="valid",
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                            bias_initializer=keras.initializers.Constant(0.1), kernel_regularizer=l2(0.001))(layer_2)
    layer_4 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu",padding="valid",strides=(1, 1),
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                            bias_initializer=keras.initializers.Constant(0.1), kernel_regularizer=l2(0.001))(layer_3)
    layer_5 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu",padding="valid",strides=(1, 1),
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                            bias_initializer=keras.initializers.Constant(0.1), kernel_regularizer=l2(0.001))(layer_4)

    layer_6 = layers.Flatten()(layer_5)

    layer_7 = layers.Dense(1164, activation="relu",
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=keras.initializers.Constant(0.1), kernel_regularizer=l2(0.001))(layer_6)
    layer_8 = layers.Dense(100, activation="relu", kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=keras.initializers.Constant(0.1), kernel_regularizer=l2(0.001))(layer_7)
    layer_9 = layers.Dense(50, activation="relu", kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=keras.initializers.Constant(0.1), kernel_regularizer=l2(0.001))(layer_8)
    layer_10 = layers.Dense(10, activation="relu", kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                            bias_initializer=keras.initializers.Constant(0.1), kernel_regularizer=l2(0.001))(layer_9)
    output_tensor = layers.Dense(1,kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                                 bias_initializer=keras.initializers.Constant(0.1), kernel_regularizer=l2(0.001))(layer_10)

    model = keras.Model(input_tensor, output_tensor)
    return model


x_train,y_train = data_extraction.x_train,data_extraction.y_train
x_test,y_test = data_extraction.x_test,data_extraction.y_test


class Data_generator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size=100):
       # print(" herer")
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.point=0

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        if(self.point+self.batch_size) > len(self.x):
            self.point = 0
        batch_x = self.x[self.point:(self.point+self.batch_size)]
        batch_y = self.y[self.point:(self.point+self.batch_size)]
        self.point += self.batch_size
        images = self.get_images(batch_x)
        return np.array(images),np.array(batch_y)

    def get_images(self, batch_x):
        x_ =[]
        for i in batch_x:
            x_.append(cv2.resize(cv2.imread(os.path.join("driving_dataset", i))[-150:],(200, 66)) / 255.0)
        return x_


train_generator = Data_generator(x_train,y_train)

test_generator = Data_generator(x_test,y_test)


model = get_model()
model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

model.fit_generator(generator=train_generator,epochs=30)
#model.save_weights("weightsfinal")
model.save("testing_model")




