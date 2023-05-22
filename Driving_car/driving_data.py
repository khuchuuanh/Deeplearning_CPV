import cv2
import random
import numpy as np
import csv
xs = []
ys = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

#read data.txt
with open("driving_data/steering angle.csv") as f:
    reader = csv.reader(f)
    for line in reader:
        xs.append("driving_data/driving_dataset/data/" + line[0])
        ys.append(float(line[1]) * 3.14159265 / 180)

#get number of images
num_images = len(xs)
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

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for  i in range(0, batch_size):
        x_out.append(cv2.resize(cv2.imread(f'{train_xs[(train_batch_pointer + i) % num_train_images]}.jpg')[-150:], (200, 66)) / 255.0)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])

    train_batch_pointer +=  batch_size
    return  x_out, y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(cv2.resize(cv2.imread(f'{val_xs[(val_batch_pointer + i) % num_val_images]}.jpg')[-150:], (200, 66)) / 255.0)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out



