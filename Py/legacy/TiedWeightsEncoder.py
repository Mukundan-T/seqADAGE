"""
Georgia Doing 2020
TiedWeightsEncoder class

A class for a tied-weight autoencoder.

Usage
	constructor

"""
#import os
##os.environ['KERAS_BACKEND'] = 'theano'
#from keras.layers import Input, Dense, Activation, Layer
#from keras import activations
#import keras.backend as K


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, regularizers, layers, initializers, models, activations
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


class TiedWeightsEncoder(Layer):

	"""
	Encoder - Decoder tied weights
	"""

	def __init__(self, output_dim, encoded, activation="sigmoid",  **kwargs): #
		self.output_dim = output_dim
		self.encoder = encoded
		#self.activation = "tanh" #activations.get(activation)
		super(TiedWeightsEncoder, self).__init__(**kwargs)

	def build(self): #, input_shape
		print("build")
		self.kernel = K.transpose(self.encoder.kernel) #self.encoder.weights
		#self._kernel = self.kernel
		print("kernel set")
		self.bias = self.add_weight(
            shape=(self.output_dim,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )
		print("bias set")
		print(self.kernel.dtype)
		print(self.encoder.weights[1].dtype)
		print(self.encoder.weights[0].dtype)
		#super(TiedWeightsEncoder, self).build() #input_shape

	def compute_output_shape(self, input_shape):
		return input_shape

	def call(self, x):
		if not training:
            #print("not training")
			return self.kernel
		print("x dt: " + str(x.dtype))
		#print(x[0:3][0:3])
		print("kern dt: " + str(self.kernel.dtype))
		#print(self.kernel[0:3][0:3])
		print("b dt: " + str(self.bias.dtype))
		#print(self.bias[0:3])
		#output = tf.matmul(x - self.encoder.weights[1], tf.transpose(self.encoder.weights[0]), name = 'dot')
		#print(output)i
		#output = K.dot(x, self.kernel) + self.bias
		output = self.kernel
		#print(output[0:3][0:3])
		#if self.activation is not None:
		#	output = self.activation(output)
		print("output (just kernel) dt: " + str(output))
		return output

	#def compute_output_shape(self): #, input_shape
	#	return input_shape[0], self.output_dim #input_shape[0], 
