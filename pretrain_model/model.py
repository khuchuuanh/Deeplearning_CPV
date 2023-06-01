import tensorflow as tf
import matplotlib.pyplot  as plt
import os
import  tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import  image_dataset_from_directory


# Data processing

train_dataset = image_dataset_from_directory('cats_and_dogs/train',
                                             shuffle  = True,
                                             batch_size = 256,
                                             image_size = (160,160))  

validation_dataset = image_dataset_from_directory('cats_and_dogs/validation',
                                                  shuffle  = True,
                                                  batch_size = 512,
                                                  image_size = (160,160))

print(train_dataset.class_names) # return the number of image and the name of subfodler
print(validation_dataset.class_names)


for images, labels in train_dataset:
  print(images.shape)
  print(labels.shape)
  break
  

for images, labels in train_dataset.take(1): # take function will return the first thing of the train_dataset
  print(images.shape)
  print(labels.shape)


class_names = train_dataset.class_names
plt.figure(figsize = (15,15))
for images, labels in train_dataset.take(1):
  # images : (512,160,160,3)
  for i in range(16):
    ax =plt.subplot(4,4, i+1)
    plt.imshow(images[i].numpy().astype('unit8'))  # hình ảnh đang ở dưới dạng tensor => phải chuyển qua dạng numpy vì plt chỉ tương tác với numpy
                                                  # nếu ở dưới dạng tensor thì nó sẽ dùng gpu còn dạng numpy sẽ dùng cpu
    plt.title(class_names[labels[i]])
    plt.axis('off')


# Data Processing layer 
# data_augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'), # flip : lat anh
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)])

# training
for  images, _ in train_dataset:
  augmented_image = data_augmentation(images)

  data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, offset = 0.)])

# training
for images, _  in train_dataset.take(1):
  # show images
  plt.figure(figsize = (18,15))
  for i in range(6):
    augmented_image = data_augmentation(images[0:1])
    ax  = plt.subplot(1,6 + i)
    plt.imshow(augmented_image[0])
    plt.axis('off')