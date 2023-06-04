import tensorflow as tf
import matplotlib.pyplot  as plt
import os
import  tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import  image_dataset_from_directory

# Visualize data




# Data processing

train_dataset = image_dataset_from_directory('/content/drive/MyDrive/pre_train/dog_cat/train/',
                                             shuffle = True,
                                             batch_size = 256,
                                             image_size = (160,160))

validation_dataset = image_dataset_from_directory('/content/drive/MyDrive/pre_train/dog_cat/val/',
                                                  shuffle = True,
                                                  batch_size = 256,
                                                  image_size = (160,160))

print(train_dataset.class_names)
print(validation_dataset.class_names)


for images, labels in train_dataset:
  print(images.shape)
  print(labels.shape)
  break
  

for images, labels in train_dataset.take(1): # take function will return the first thing of the train_dataset
  print(images.shape)
  print(labels.shape)


class_names = train_dataset.class_names
plt.figure(figsize = (10,10))
for images, labels in train_dataset.take(1):
  # images : (64,250,250,3)
  for i in range(16):
    ax =plt.subplot(4,4, i+1)
    plt.imshow(images[i].numpy().astype('unint8'))  # hình ảnh đang ở dưới dạng tensor => phải chuyển qua dạng numpy vì plt chỉ tương tác với numpy
                                                  # nếu ở dưới dạng tensor thì nó sẽ dùng gpu còn dạng numpy sẽ dùng cpu
    plt.title(class_names[labels[i]])
    plt.axis('off')


# train model from scratch

# Load model
model1 = tf.keras.applications.VGG16(input_shape = (160,160,3),
                                    include_top = False,
                                    weights = None)

model1.summary()

# data augmentation

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset = -1)
])

# flattening

global_layer = tf.keras.layers.GlobalMaxPooling2D()

# final layer
prediction_layer = tf.keras.layers.Dense(1)

# Construct a new network

inputs = tf.keras.Input(shape = (160,160,3))
x = data_augmentation(inputs)
x = model1(x)
x = global_layer(x)
outputs = prediction_layer(x)
model1 = tf.keras.Model(inputs, outputs)


print(type(model1.layers))
print(len(model1.layers))

for layer in model1.layers:
  print(layer.name, '_', layer.trainable)


model1.compile(loss = keras.losses.BinaryCrossentropy( from_logits = True),
               optimizer =keras.optimizers.Adam(lr = 0.0001),
               metrics = ['Accuracy'])

history_fine = model1.fit(train_dataset, epochs = 50, validation_data =validation_dataset)



plt.plot(history_fine.history['loss'], label = 'train_loss')
plt.plot(history_fine.history['val_loss'], label = 'val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()


plt.plot(history_fine.history['Accuracy'], label = 'train_Accuracy')
plt.plot(history_fine.history['val_Accuracy'], label = 'val_Accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Tranfer learning

model2 = tf.keras.applications.VGG16(input_shape =(160,160,3),
                                     include_top = False,
                                     weights = 'imagenet')

model2.trainable = False

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset = -1)
])

global_layer = tf.keras.layers.GlobalMaxPooling2D()

prediction_layer = tf.keras.layers.Dense(1)

# Contruct a new network

inputs = tf.keras.Input(shape = (160,160,3))
x = data_augmentation(inputs)
x = model2(x)
x = global_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model2 = tf.keras.Model(inputs, outputs)

model2.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = True), # from_logits = True: nếu đầu ra là 1 node
                      optimizer = tf.keras.optimizers.Adam(lr = 0.0001),
               metrics =['accuracy'])

history_fine = model2.fit(train_dataset, epochs = 50, validation_data= validation_dataset)

plt.plot(history_fine.history['loss'], label = 'train_loss')
plt.plot(history_fine.history['val_loss'], label = 'val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(history_fine.history['accuracy'], label = 'train_accuracy')
plt.plot(history_fine.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Fine tuning

model3  = tf.keras.applications.VGG16(input_shape = (160,160,3), 
                                    include_top = False,
                                    weights = 'imagenet')

model3.summary()

# freeze some first layer
fine_tune_at = 14
for layer in model3.layers[:14]:
  layer.trainable = False

# processing data

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset = -1)
])

# flattening
global_layer = tf.keras.layers.GlobalMaxPooling2D()

# final layer

prediction_layer = tf.keras.layers.Dense(1)

# Construct a new network

inputs = tf.keras.Input(shape =(160,160,3))
x = data_augmentation(inputs)
x =  model3(x)
x = global_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model3 = tf.keras.Model(inputs, outputs)

model3.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
                      optimizer = tf.keras.optimizers.Adam(lr = 0.0001),
               metrics =['accuracy'])

history_fine3 = model3.fit(train_dataset, epochs = 50, validation_data= validation_dataset)

plt.plot(history_fine3.history['loss'], label = 'train_loss')
plt.plot(history_fine3.history['val_loss'], label = 'val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

plt.plot(history_fine3.history['accuracy'], label = 'train_accuracy')
plt.plot(history_fine3.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Initial VGG16 wieght for small data

model4  = tf.keras.applications.VGG16(input_shape = (160,160,3), 
                                    include_top = False,
                                    weights = 'imagenet')

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset = -1)
])

# flattening
global_layer = tf.keras.layers.GlobalMaxPooling2D()

# final layer

prediction_layer = tf.keras.layers.Dense(1)

# Construct a new network

inputs = tf.keras.Input(shape =(160,160,3))
x = data_augmentation(inputs)
x =  model4(x)
x = global_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model4 = tf.keras.Model(inputs, outputs)

model4.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = True), 
                      optimizer = tf.keras.optimizers.Adam(lr = 0.0001),
               metrics =['accuracy'])

history_fine4 = model4.fit(train_dataset, epochs = 50, validation_data= validation_dataset)

plt.plot(history_fine4.history['loss'], label = 'train_loss')
plt.plot(history_fine4.history['val_loss'], label = 'val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(history_fine4.history['accuracy'], label = 'train_accuracy')
plt.plot(history_fine4.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()