import cv2
import random
import numpy as np
import csv
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

