import tensorflow as tf
import tensorflow.keras as keras
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import keras_tuner as kt
import numpy as np
import random 

(train_images, train_labels),(test_images, test_labels) = cifar10.load_data()
print(train_images.shape)
print(test_images.shape)

# Standard data
train_images = train_images/ 255.0
test_images = test_images/ 255.0

train_size = train_images.shape[0]
random_num = random.randint(0, train_size)
print(f"%dth image of the dataset" %random_num)
plt.imshow(train_images[random_num])
plt.show()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize = (10,10))
for i in range(25):
  plt.subplot(5,5, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap = plt.cm.binary)
  plt.xlabel(class_names[train_labels[i][0]])
plt.show()

checkpoint_path = "./checkpoint_file/model_checkpoint.ckpt"
checkpoint_callback  = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weight_only = True,
    save_best_only = True,
    monitor = 'val_loss',
    verbose = 1
)

# Building model

model =  keras.models.Sequential()
model.add(tf.keras.Input(shape = (32,32,3)))

model.add(keras.layers.Conv2D(64, 3, 1, padding = 'same', activation = 'relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(64, 3, 1, padding = 'same', activation = 'relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.MaxPooling2D(2))

model.add(keras.layers.Conv2D(128,3,1, padding = 'same', activation = 'relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(128,3,1, padding = 'same' ,activation = 'relu')) 
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.MaxPooling2D(2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation = 'relu')) 
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.summary()

model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics = ['Accuracy']
)

history_data = model.fit(train_images, train_labels,
                         validation_data = (test_images, test_labels),
                         batch_size = 512, epochs = 50, callbacks = [checkpoint_callback])

plt.plot(history_data.history['loss'], label = 'train_loss')
plt.plot(history_data.history['val_loss'], label= 'val_loss')
plt.xlabel('epoch')
plt.ylabel("loss")
plt.legend()
plt.show()

plt.plot(history_data.history['Accuracy'], label = 'train_Accuracy')
plt.plot(history_data.history['val_Accuracy'], label = 'val_Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()