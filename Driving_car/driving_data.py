import cv2
import random
import numpy as np
import csv
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array



xs = []
ys = []

train_batch_pointer = 0
val_batch_pointer = 0
with open("driving_data/steering angle.csv") as f:
    reader = csv.reader(f)
    for line in reader:
        xs.append("driving_data/" + line[0])
        ys.append(float(line[1]) * 3.14159265 / 180)

num_images = len(xs)
#print("num_images : ", num_images)
#shuffle list of images
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def read_and_process_images(folder_path, target_shape):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            try:
                image = Image.open(image_path)
                image = image.resize(target_shape[:2])
                image_array = img_to_array(image)
                images.append(image_array)
            except Exception as e:
                print(f"Error reading image '{filename}': {str(e)}")
    
    images = np.array(images) 
    return images

folder_path = "driving_data/driving_dataset"
target_shape = (66, 200, 3)
images = read_and_process_images(folder_path, target_shape)
images = images/255.0


