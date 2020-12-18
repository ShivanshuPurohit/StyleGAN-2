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
	

def upsample_to size(x):
	y = im_size / x.shape[2]
	x = K.resize_immages(x, y, y, 'channels_last', interpolation='bilinear')
	return x


#Blocks
def g_block(inp, istyle, inoise, fil, u=True):
	if u:
		#custom upsampling
		out = Lambda(upsample, output_shape=[None, inp.shape[2]*2, inp.shape[2]*2, None])(inp)
	else:
		out = Activation('linear')(inp)

	rgb_style = Dense(fil, kernel_initializer=VarianceScaling(200/out.shape[2]))(istyle)
	style = Dense(inp.shape[-1], kernel_initializer='he_uniform')(istyle)
	delta = Lambda(crop_to_fit)([inoise, out])
	d = Dense(fil, kernel_initializer='zeros')(delta)

	out = Conv2DMod(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_uniform')([out, style])
	out = add([out, d])
	out = LeakyReLU(0.2)(out)

	style = Dense(fil, kernel_initializer='he_uniform')(istyle)
	d = Dense(fil, kernel_initializer='zeros')(delta)
	out = Conv2DMod(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_uniform')([out, style])
	out = add([out, d])
	out = LeakyReLU(0.2)(out)

	return out, to_rgb(out, rgb_style)


def d_block(inp, fil, p=True):
	res = Conv2D(fil, 1, kernel_initializer='he_uniform')(inp)
	out = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_uniform')(inp)
	out = LeakyReLU(0.2)(out)
	out = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_uniform')(out)
	out = LeakyReLU(0.2)(out)
	out = add([res, out])

	if p:
		out = AveragePooling2D()(out)
	return out


def to_rgb(inp, style):
	size = inp.shape[2]
	x = Conv2D(3, 1, kernel_initializer=VarianceScaling(200/size), demod=False)([inp, style])
	return Lambda(upsample_to_size, output_shape=[None, im_size, im_size, None])(x)


def from_rgb(inp, conc=None):
	fil = int(im_size*4/inp.shape[2])
	z = AveragePooling2D()(inp)
	x = Conv2D(fil, 1, kernel_initializer='he_uniform')(z)
	if conc is not None:
		x = concatenate([x, conc])
	return x, z


class GAN(object):
	def __init__(self, steps=1, lr=0.0001, decay=0.00001):
		self.D = None
		self.S = None
		self.G = None
		self.GE = None
		self.SE = None
		self.DM = None
		self.AM = None
		self.lr = lr
		self.steps = steps
		self.beta = 0.999

		self.discriminator()
		self.generator()

		self.GMO = Adam(lr=self.lr, beta_1=0, beta_2=0.999)
		self.DMO = Adam(lr=self.lr, beta_1=0, beta_2=0.999)

		self.GE = clone_model(self.G)
		self.GE.set_weights(self.G.get_weights())
		self.SE = clone_model(self.S)
		self.SE.set_weights(self.S.get_weights())

	def discriminator(self):
		if self.D:
			return self.D
		inp = Input(shape=[im_size, im_size, 3])
		x = d_block(inp, 1*cha)
		x = d_block(x, 2*cha)
		x = d_block(x, 4*cha)
		x = d_block(x, 6*cha)
		x = d_block(x, 8*cha)
		x = d_block(x, 16*cha)
		x = d_block(x, 32*cha, p=False)
		x = Flatten()(x)
		x = Dense(1, kernel_initializer='he_uniform')(x)
		self.D = Model(inputs=inp, outputs = x)
		return self.D

	def generator(self):
		if self.G:
			return self.G

		self.S = Sequential()
		self.S.add(Dense(512, input_shape=[latent_size]))
		self.S.add(LeakyReLU(0.2))
		self.S.add(Dense(512))
		self.S.add(LeakyReLU(0.2))
		self.S.add(Dense(512))
		self.S.add(LeakyReLU(0.2))
		self.S.add(Dense(512))
		self.S.add(LeakyReLU(0.2))

		inp_style = []
		for i in range(n_layers):
			inp_style.append(Input([512]))
		inp_noise = Input([im_size, im_size, 1])

		#latent
		x = Lambda(lambda x: x[:, :1]*0+1)(inp_style[0])
		outs = []

		#the model
		x = Dense(4*4*4*cha, activation='relu', kernel_initializer='random_normal')(x)
		x = Reshape([4, 4, 4*cha])(x)
		x, r = g_block(x, inp_style[0], inp_noise, 32*cha, u=False)
		outs.append(r)
		x, r = g_block(x, inp_style[1], inp_noise, 16*cha)
		outs.append(r)
		x, r = g_block(x, inp_style[2], inp_noise, 8*cha)
		outs.append(r)
		x, r = g_block(x, inp_style[3], inp_noise, 6*cha)
		outs.append(r)
		x, r = g_block(x, inp_style[4], inp_noise, 4*cha)
		outs.append(r)
		x, r = g_block(x, inp_style[5], inp_noise, 2*cha)
		outs.append(r)
		x, r = g_block(x, inp_style[6], inp_noise, 1*cha)
		outs.append(r)
		x = add(outs)

		x = Lambda(lambda y: y/2+0.5)(x) #for a good init
		self.G = Model(inputs=inp_style+[inp_noise], outputs=x)
		return self.G

	def GanModel(self):
		inp_style = []
		style = []
		for i in range(n_layers):
			inp_style.append(Input([latent_size]))
			style.append(self.SE(inp_style[-1]))
		inp_noise = Input([im_size, im_size, 1])
		gf = self.GE(style+[inp_noise])
		self.GMA = Model(inputs=inp_style+[inp_noise], outputs=gf)
		return self.GMA

	def EMA(self):
		for i in range(len(self.G.layers)):
			up_weight = self.G.layers[i].get_weights()
			old_weight = self.GE.layers[i].get_weights
			new_weight = []
			for j in range(len(up_weight)):
				new_weight.append(old_weight[j]*self.beta+(1-self.beta)*up_weight[j])
			self.GE.layers[i].set_weights(new_weight)

		for i in range(len(self.S.layers)):
			up_weight = self.S.layers[i].get_weights()
            old_weight = self.SE.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1-self.beta) * up_weight[j])
            self.SE.layers[i].set_weights(new_weight)


    def MAinit(self):
    	self.GE.set_weights(self.G.get_weights())
    	self.SE.set_weights(self.S.get_weights())
