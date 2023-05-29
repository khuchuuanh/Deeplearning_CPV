import tensorflow as tf
import pandas as pd
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import cv2
from subprocess import call


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


# building model

model = keras.models.Sequential()
model.add(tf.keras.Input(shape = (66,200,3)))

model.add(keras.layers.BatchNormalization())


model.add(keras.layers.Conv2D(24,5,2 ))

model.add(keras.layers.Conv2D(36,5,2))

model.add(keras.layers.Conv2D(48,5,2))

model.add(keras.layers.Conv2D(64,3,1))

model.add(keras.layers.Conv2D(64,3,1))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(1164))
model.add(keras.layers.Dropout(0.8))

model.add(keras.layers.Dense(100))
model.add(keras.layers.Dropout(0.8))

model.add(keras.layers.Dense(50))
model.add(keras.layers.Dropout(0.8))

model.add(keras.layers.Dense(10))
model.add(keras.layers.Dropout(0.8))

model.add(keras.layers.Dense(1))

model.summary()


callback_path = './checkfolder/checkpoint.ckpt'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    callback_path,
    save_weights_only= True,
    save_best_only= True,
    monitor= "val_loss",     # theo dõi và quyết định  lưu checkpoint tốt nhất
    verbose= 1
) 


model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              metric = ["Accuracy"]
              )


history_data = model.fit(train_images, train_labels,
                         validation_data = (test_images, test_labels),
                         batch_size = 100 ,
                         epochs = 30,
                         callbacks = [checkpoint_callback]
                         )



windows = False
if os.name == "nt":
    windows = True

sess  = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, './save/model.ckpt')
img = cv2.imread('./steering_wheel_image.jpg',0)
rows, cols = img.shape

smoothed_angle = 0

cap =cv2.VideoCapture(0)
while(cv2.waitKey(10) != ord('q')):
    ret, frame = cap.read()
    image = cv2.resize(frame, (200, 66))/255.0
    degrees = model.y.eval(feed_dict = {model.x: [image], model.keep_prob:1.0})[0][0] * 180/3.14159265
    if not windows:
        call('clear')
    print("Predicted steering angle : " + str(degrees) + "degrees")
    cv2.imshow('frame', frame)

    smoothed_angle += 0.2*pow(abs((degrees - smoothed_angle)),2.0/3.0) *(degrees - smoothed_angle)/ abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), -smoothed_angle,1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow("steering wheel", dst)

cap.release()
cv2.destroyAllWindows()


