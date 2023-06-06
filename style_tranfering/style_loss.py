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


style_path = ('/content/drive/MyDrive/style tranfer/style4.jpg')
style_image = load_img(style_path)
imshow(style_image[0], 'style_image')
print(style_image.shape)

def get_vgg_layers(layer_names):
  vgg = tf.keras.applications.VGG19(include_top = False,
                                    weights = 'imagenet')
  
  vgg.trainable  = False
  outputs = [vgg.get_layer(name).output for name in layer_names]
  model = tf.keras.Model([vgg.input], outputs)
  return model


style_layers = ['block5_conv1']
my_net = get_vgg_layers(style_layers)

def compute_feature(inputs):
  inputs = inputs*255.0 
  preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs) 
  style_output = my_net(preprocessed_input)
  return style_output

style_targets = compute_feature(style_image) 

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

