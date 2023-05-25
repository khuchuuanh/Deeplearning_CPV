import tensorflow as tf
import pandas as pd
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

label_data = pd.read_csv('driving_data/steering angle.csv')

images = []
labels = []

for index, row in label_data.iterrows():
    image_path = ('driving_data/driving_dataset/data/'+  str(int(row['image'])) +'.jpg')  # Update 'image_folder' with the path to your image folder
    image = Image.open(image_path)
    image = image.resize((66, 200))  # Resize the image to (66, 200)
    image = tf.keras.preprocessing.image.img_to_array(image)
    images.append(image)
    labels.append(row['steering angle'])

images = np.array(images)
labels = np.array(labels)

print(images.shape)
print(labels.shape)

images = images / 255.0

train_images, test_images, train_labels, test_labels = train_test_split(images,labels, test_size= 0.2)

train_images = np.array(train_images)
test_images = np.array(test_images)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

train_images = train_images.reshape((-1, 66,200,3))
test_images = test_images.reshape((-1, 66,200,3))





