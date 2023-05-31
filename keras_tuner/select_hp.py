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

# building model for keras tuner

def build_model(hp):
  model =  keras.models.Sequential()
  model.add(tf.keras.Input(shape = (32,32,3)))

  filters = hp.Int('filters', min_value  = 32, max_value = 128, step = 32)
  kernel_size = hp.Choice("kernel_size", values = [3,5])
  hp_unit = hp.Int('units', min_value = 32, max_value = 512, step = 32)
  hp_learning_rate = hp.Choice("learning_rate", values = [1e-2, 1e-3, 1e-4])

  model.add(keras.layers.Conv2D(filters, kernel_size, 1, padding = 'same', activation = 'relu'))
  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.BatchNormalization())

  model.add(keras.layers.Conv2D(filters,kernel_size,1, padding = 'same' ,activation = 'relu'))
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

  model.add(keras.layers.Dense(hp_unit, activation = 'relu'))

  model.add(keras.layers.Dense(10, activation = 'softmax'))

  model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp_learning_rate),
                loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics = ["Accuracy"])
  
  return model

  # các tham số hiệu chỉnh
  # filters = hp.Int('filters', min_value  = 32, max_value = 128, step = 32)
  # kernel_size = hp.Choice("kernel_size", values = [3,5])
  # hp_unit = hp.Int('units', min_value = 32, max_value = 512, step = 32)
  # hp_learning_rate = hp.Choice("learning_rate", values = [1e-2, 1e-3, 1e-4])

tuner = kt.Hyperband(build_model,
                     objective='val_Accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='./keras_tuner1',
                     project_name='Project_2')
  
tuner.search_space_summary()

stop_early = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5)
tuner.search(train_images, train_labels, epochs=50, validation_split=0.2, callbacks=[stop_early])
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print('The best unit is : ',best_hps.get("units"))
print("The best learning rate is :",best_hps.get("learning_rate"))
print("The best number of filters is : ",(best_hps.get("filters")))
print("The best size of kernel is :  ",(best_hps.get("kernel_size")))

model = tuner.hypermodel.build(best_hps)
history = model.fit(train_images, train_labels, epochs=50, validation_split=0.2)
val_acc_per_epoch = history.history['val_Accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

model  = tuner.hypermodel.build(best_hps)
history_data = model.fit(train_images, train_labels, epochs=best_epoch, validation_split=0.2)

# train loss and validation loss visualization after using selected HP
plt.plot(history_data.history['loss'], label = 'train_loss')
plt.plot(history_data.history['val_loss'], label = 'val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(history_data.history['Accuracy'], label = 'train_Accuracy')
plt.plot(history_data.history['val_Accuracy'], label = 'val_Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
