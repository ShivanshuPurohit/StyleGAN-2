from PIL import image
from math import floor, log2
import numpy as np
import time
from functools import partial
from random import random
import os

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
import tensorflow as tf
import tensorflow.keras.backend as K

from datagen import dataGenerator, printProgressBar
from conv_mod import *


im_size = 256
latent_size = 512
batch_size = 16
directory = 'main'

cha = 24
n_layers = int(log2(im_size) - 1)
mixed_prob = 0.9


def noise(n):
	return np.random.normal(0.0, 0.1, size=[n, latent_size]).astype('float32')


def noiseList(n):
	return [noise(n)] * n_layers


def mixedList(n):
	tt = int(random() * n_layers)
	p1 = [noise(n)] * tt
	p2 = [noise(n)] * (n_layers - tt)
	return p1 + [] + p2


def nImage(n):
	return np.random.uniform(0.0, 1.0, size=[n, im_size, im_size, 1]).astype('float32')


#loss functions
def gradient_penalty(samples, output, weight):
	gradients = K.gradients(output, samples)[0]
	gradients_sqr = K.square(gradients)
	gradient_penalty = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
	return K.mean(gradient_penalty) * weight


def hinged(y_true, y_pred):
	return K.mean(K.relu(1.0+(y_pred*y_true)))


def w_loss(y_true, y_pred):
	return K.mean(y_true*y_pred)


def crop_to_fit(x):
	height = x[1].shape[1]
	width = x[1].shape[2]
	return x[0][:, :height, :width, :]


def upsample(x):
	return K.resize_immages(x, 2, 2, "channels_last", interpolation='bilinear')
	
