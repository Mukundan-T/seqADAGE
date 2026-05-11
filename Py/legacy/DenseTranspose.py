# from http://14.139.161.31/OddSem-0822-1122/Hands-On_Machine_Learning_with_Scikit-Learn-Keras-and-TensorFlow-2nd-Edition-Aurelien-Geron.pdf pg. 577

from keras.layers import Layer, Dense
import tensorflow as tf
import keras
from keras import ops
#import numpy

tf.compat.v1.enable_eager_execution()
print(tf.executing_eagerly())

keras.saving.get_custom_objects().clear()
@keras.saving.register_keras_serializable() #package="dla"
class DenseTranspose(keras.layers.Layer):
    def __init__(self, dense,  activation=None, **kwargs):
        self.dense = dense
        #self.layer_size = layer_size
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)
    
    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias", initializer="zeros",
            shape=[self.dense.get_weights()[0].shape[0]]) # input_shape  self.dense.input.shape[-1]
        super().build(batch_input_shape)
    
    def call(self, inputs, training=None):
        #t_weights = self.dense.get_weights()
        z = tf.matmul(inputs, self.dense.kernel, transpose_b=True)
        return self.activation(z + self.biases)

    def get_config(self):
        config = super().get_config()
        config.update({
           # "layer_size": self.layer_size,
            "dense": self.dense,
            #"activation": self.activation,
        })
        return config

    @classmethod
    def from_config(cls, config):

       # config["layer_size"] = keras.layers.deserialize(config["layer_size"])
        config["dense"] = keras.layers.deserialize(config["dense"])
        #config["activation"] = keras.layers.deserialize(config["activation"])
        return cls(**config)