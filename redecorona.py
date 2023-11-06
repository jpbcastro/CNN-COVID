import os
import paths
import telebot
import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split

#treinamento da rede neural
global img_size
img_size = (75,75)

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    paths.data_path,
    labels='inferred',
    label_mode='categorical',
    class_names=['COVID','Infection','Normal'],
    color_mode='rgb',
    batch_size=600,
    image_size=img_size,
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='both',
    interpolation='bilinear'
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model=tf.keras.Sequential()
model.add(tf.keras.layers.Rescaling(1./255, input_shape=(75, 75, 3)))
model.add(tf.keras.layers.Conv2D(8, (3,3), activation='relu'))
model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid"))
model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

model.fit(train_ds, epochs=30, validation_data=(val_ds))

model.save(paths.model_path)