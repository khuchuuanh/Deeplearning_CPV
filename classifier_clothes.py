from sklearn.metrics import classification_report
import tensorflow as tf
import tensorflow.keras as keras
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import random
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_images.shape)
print(test_images.shape)

train_images = train_images / 255.0
test_images = test_images/ 255.0

train_images = tf.reshape(train_images, (60000, 28,28,1))
test_images = tf.reshape(test_images, (10000, 28,28,1))

# Class of data
diction = {0: 'T-shirt/top', 1: 'Trouser',
            2: 'Pullover',3: 'Dress', 4: 'Coat', 5: 'Sandal',
           6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Anke boot',           
           }


# Data visualization
train_size = train_images.shape[0]
random = random.randint(0,6000)
print(random)
plt.imshow(train_images[random])
plt.show()

L = 7
W = 7
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel() 
for i in np.arange(0, L * W):  
    axes[i].imshow(train_images[i],cmap='gray')
plt.subplots_adjust(wspace=1)


# Build model to train data
model = keras.models.Sequential()
model.add(tf.keras.Input(shape = (28,28,1)))
model.add(keras.layers.Conv2D(64,3,1, padding = 'same' ,activation = 'relu')) # parmeter = 64((3*3*1)+1)
model.add(keras.layers.BatchNormalization()) # y_bar = gamma*Xi + Beta
model.add(keras.layers.Conv2D(64,3,1, padding = 'same' ,activation = 'relu')) # parmeter = 64((3*3*64)+1)
model.add(keras.layers.BatchNormalization()) # 
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(128,3,1, padding = 'same', activation = 'relu'))# parmeter = 128((3*3*64)+1)
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(128,3,1, padding = 'same' ,activation = 'relu')) # parmeter = 128((3*3*128)+1)
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation = 'relu')) # parmeter =  128 *(flatten +1) 
model.add(keras.layers.Dense(10, activation = 'softmax')) # parmeter = 10 (128+1) 

model.summary()


model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics = ['Accuracy'])
history_data = model.fit(train_images, train_labels,
                         validation_data = (test_images, test_labels),
                         batch_size = 512, epochs = 100)


# train loss and validation loss visualization
plt.plot(history_data.history['loss'], label = 'train_loss')
plt.plot(history_data.history['val_loss'], label = 'val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# Accuracy visualization
plt.plot(history_data.history['Accuracy'], label = 'train_Accuracy')
plt.plot(history_data.history['val_Accuracy'], label = 'val_Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

img = plt.imread('test.png')
plt.imshow(img)
plt.show()

img = load_img('test.png', grayscale=True, target_size=(28, 28))
img = img_to_array(img)
img = img.reshape(1, 28, 28, 1)
img = img / 255.0
predictions = np.argmax(model.predict(img),axis=1)
predict = int(predictions[0])
print('This image belongs to class : ', predict , diction[predict])
print(diction)

predicted_classes = model.predict(test_images)
predicted_classes = np.argmax(predicted_classes , axis = 1)

# Confusion matrix
labels = [0, 1, 2,3,4,5,6,7,8,9]
cm = confusion_matrix(test_labels, predicted_classes)
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap=plt.cm.Reds)
ax.set_xticks(labels)
ax.set_yticks(labels)
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel('Actual')
ax.set_ylabel('Prediction')
ax.set_title('Confusion Matrix')
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
# Thêm colorbar
cbar = ax.figure.colorbar(im, ax=ax)
plt.show()

cm = confusion_matrix(test_labels,predicted_classes, labels=labels)
sns.heatmap(cm,
            annot=True)
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()

# classification report 
print(classification_report(test_labels, predicted_classes))