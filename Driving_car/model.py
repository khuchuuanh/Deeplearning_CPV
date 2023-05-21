import  tensorflow as tf
import tensorflow.keras as keras

# building model

model = keras.models.Sequential()
model.add(tf.keras.Input(shape = (66,200,3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(3,(5,5),1, padding = "same"))
model.add(keras.layers.Conv2D(24, (5,5),2))
model.add(keras.layers.Conv2D(36, (5,5),2))
model.add(keras.layers.Conv2D(48, (5,5),2))
model.add(keras.layers.Conv2D(64, (3,3),1))
model.add(keras.layers.Conv2D(64, (3,3),1))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1164))
model.add(keras.layers.Dense(100))
model.add(keras.layers.Dense(50))
model.add(keras.layers.Dense(10))
model.summary()