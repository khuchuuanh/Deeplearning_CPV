import os 
import tensorflow as tf
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import  PIL.Image
from PIL import Image
import time
import functools

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype = np.uint8)
  return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels = 3)
  img = tf.image.convert_image_dtype(img, tf.float32) #[0,1]

  img = tf.image.resize(img,(422, 512))
  img = img[tf.newaxis, :] #(bs,H,W,C)
  return img


# show image
def imshow(image, title = None):
  plt.imshow(image)
  if title:
    plt.title(title)


content_path = ('/content/drive/MyDrive/style tranfer/content.jpg')
content_image = load_img(content_path)
imshow(content_image[0], 'content image')
print(content_image.shape)


def get_vgg_layers(layer_names):
  vgg = tf.keras.applications.VGG19(include_top = False,
                                    weights = 'imagenet')
  
  vgg.trainable  = False
  outputs = [vgg.get_layer(name).output for name in layer_names]
  model = tf.keras.Model([vgg.input], outputs)
  return model

# create the model:

content_layers = ['block5_conv1']
my_net = get_vgg_layers(content_layers)

def compute_feature(inputs):
  inputs = inputs*255.0 # vì hình input của VGG là ảnh chạy từ khoảng 0 -255 nên phải nhân vs 255
  preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs) # preprocess_input: dùng để nomarlize data
  content_outputs = my_net(preprocessed_input)
  return content_outputs

# compute  for target image
content_targets = compute_feature(content_image) # feature map  for target


image = tf.Variable(tf.random.uniform(content_image.shape, 
                                      minval= 0,
                                      maxval = 1))

print(image.shape)

def clip_0_1(image):
  return tf.clip_by_value(image, 
                          clip_value_min = 0.0, # khi đạo hàm ảnh sẽ có giá trị nằm ngoài khoảng 0,1 nên cần dùng hàm này để khử dữ liệu này
                          clip_value_max = 1.0) # những dữ liệu nằm ngoài khoảng 0,1 là những dữ liệu sai.
                                                # hàm này được dùng sau khi giá trị đạo hàm được cập nhật

display.display(tensor_to_image(image[0]))

opt = tf.optimizers.Adam(learning_rate = 0.01,
                         beta_1= 0.99,
                         epsilon = 1e-1)

# content loss

def content_loss(content_ouputs):
  loss = tf.reduce_mean((content_ouputs - content_targets)**2)
  return loss


@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    # compute f1 and f2
    outputs = compute_feature(image)

    #compute (f1-f2)**2
    loss = content_loss(outputs)

  grad  = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))


epochs = 10
steps_per_epoch = 50

step =0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1 
    train_step(image)
    print('.', end = '')
    display.clear_output(wait = True)
    display.display(tensor_to_image(image[0]))
    print('Train step: {}'.format(step))