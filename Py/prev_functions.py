#import os
#os.environ['KERAS_BACKEND'] = 'tensorflow'
#import keras as keras

import tensorflow as tf
from tensorflow import keras

import argparse
import numpy as np
import csv
import pandas as pd

from tensorflow.keras import optimizers, regularizers, layers, initializers, models
from tensorflow.keras.layers import Input, Dense, Dropout
#from keras.models import Model, Sequential
#from tensorflow.keras import initializers
import TiedWeightsEncoder as tw
import Adage as ad
from AdageHyperModel import AdageHyperModel
from AdageHyperModelv2 import AdageHyperModelv2

from autoencoders import DenseLayerAutoencoder

from DenseTranspose import DenseTranspose

from keras.models import Model, load_model
import keras_tuner
import matplotlib.pyplot as plt
import os


#from keras import initializers

def tune_model_old(input_file, seed):
    """
	
	"""
    hp = keras_tuner.HyperParameters()
	# defining the number of neurons dynamically
    units = hp.Int(name="units", min_value=10, max_value=100, step=10)   
    # defining the dropout rate
    #dropout = hp.Int(name="dropout", min_value=0.0, max_value=0.3)
    # Automatically assign True/False values.
    act1 = hp.Choice('act1', ["sigmoid","tanh","relu", "celu"])
    act2 = hp.Choice('act2', ["sigmoid","tanh","relu", "celu"])
    shuffle = hp.Boolean("shuffle", default=False)
    init = hp.Choice('init', ["glorot_uniform","glorot_normal"]) 
    kl1 = hp.Float('kl1', min_value = 0, max_value = 1, step = 0.1)
    kl2 = hp.Float('kl2', min_value = 0, max_value = 1, step = 0.1)
    al1 = hp.Float('al2', min_value = 0, max_value = 1, step = 0.1)
    lr = hp.Float('lr', min_value = 0.001, max_value = 0.1, step = 0.01) 
    bs = hp.Int('bs', min_value = 10, max_value = 50, step = 10) 

    all_comp = pd.read_csv(input_file, index_col=0)
    gene_num = np.size(all_comp, 0)

    tuner = keras_tuner.Hyperband(
                       hypermodel=AdageHyperModel(gene_num),#
		               hyperparameters = hp,
                       objective = "val_loss", #optimize val acc
                       max_epochs=50, #for each candidate model
                       overwrite=True,  #overwrite previous results
                       directory='hyperband_search_dir', #Saving dir
                       project_name='adage_tuner')
    
    
    x_train, x_train_noisy = prep_data(all_comp, seed)
    
    np.random.seed(seed)
    train_idxs = np.random.choice(x_train.shape[0],
							      int(x_train.shape[0]*0.9), replace=False)
	#print(train_idxs[1:5])
    x_train_train = x_train[train_idxs,:]
    x_train_test = x_train[~np.in1d(range(x_train.shape[0]),train_idxs),:]

    x_train_noise_train = x_train_noisy[train_idxs,:]
    x_train_noise_test = x_train_noisy[~np.in1d(range(x_train.shape[0]),
		                                              train_idxs),:]
    
    #print(len(ss.space))
    
    tuner.search(x_train_noise_train, x_train_train,
             #max_trials=50,  # Max num of candidates to try
			 #batch_size=batch_size,
             validation_data=(x_train_noise_test,x_train_test))
    tuner.results_summary() 
    num_trials = len(tuner.oracle.trials.values())
    best_hps = tuner.get_best_hyperparameters(num_trials)
    best_models = tuner.get_best_models(num_trials)
    #ss = tuner.search_space_summary(extended=True)
    
    model = tuner.hypermodel.build(best_hps[0])
    hist  = tuner.hypermodel.fit(
	    best_hps[0], 
		model,
	    x = x_train_noise_train, 
		y = x_train_train,) 
    
    return(best_hps, tuner)


def unlinked_ae(encoding_dim, gene_num, act, init,seed, kl1, kl2, pre_w):
	input_img = layers.Input(shape=(gene_num,))
	init_s = initializers.glorot_normal(seed=seed)
	encoded = layers.Dense(encoding_dim, #input_img , #input_shape=(gene_num, )
					activation=act, #sigmoid
					#kernel_initializer = init_s,
					kernel_initializer = init,
					kernel_regularizer = regularizers.l1_l2(l1=kl1, l2=kl2),
    				activity_regularizer = regularizers.l1(0))(input_img)




	decoded = layers.Dense(gene_num, activation='sigmoid',
    				activity_regularizer = regularizers.l1(0))(encoded)

	# this model maps an input to its reconstruction
	autoencoder = models.Model(input_img, decoded)
	return(autoencoder)

def linked_ae_preT(encoding_dim, gene_num, act, init,seed, kl1, kl2):
	input_img = layers.Input(shape=(gene_num,))
	init_s = initializers.glorot_normal(seed=seed)
	encoded = layers.Dense(encoding_dim, #input_img , 
                    activation=act, #input_shape=(gene_num, )
					#kernel_initializer = init_s,
					kernel_initializer = init,
					kernel_regularizer = regularizers.l1_l2(l1=kl1, l2=kl2),
    				activity_regularizer = regularizers.l1(0))#(input_img)


	decoder = tw.TiedWeightsEncoder(input_shape=(encoding_dim,),
                                    output_dim=gene_num,
                                    encoded=encoded, 
                                    activation="sigmoid")

	autoencoder = keras.Sequential()
	autoencoder.add(pre_w)
	autoencoder.add(decoder)
	return(autoencoder)


def linked_ae(encoding_dim, gene_num, act, init,seed, kl1, kl2):
	input_img = layers.Input(shape=(gene_num,))
	init_s = initializers.glorot_normal(seed=seed)
	print("about to make encoded")
	encoded = layers.Dense(encoding_dim, #input_img ,
                    activation=act, #input_shape=(gene_num, )
					#kernel_initializer = init_s,
					kernel_initializer = init,
					kernel_regularizer = regularizers.l1_l2(l1=kl1, l2=kl2),
    				activity_regularizer = regularizers.l1(0))#(input_img)


	print("about to call tw")
	decoder = tw.TiedWeightsEncoder(#input_shape=(encoding_dim,),
                                    output_dim=gene_num,
                                    encoded=encoded, 
                                    activation=act)#(encoded) #sigmoid

	print("about to make seq")
	autoencoder = keras.Sequential()
	#autoencoder.add(input_img)
	autoencoder.add(encoded)
	autoencoder.add(decoder)
	#autoencoder = models.Model(inputs=input_img, outputs=decoded)
	return(autoencoder)

def linked_ae_ae(encoding_dim, gene_num, act, init,seed, kl1, kl2, preT=False, init_weights=[]):
	inputs = Input(shape=(gene_num,))
	d = Dropout(0.1)(inputs)
	x = DenseLayerAutoencoder([encoding_dim], activation=act, dropout=0.0,l1=0,l2=0, preT=preT, init_fun = init, act_fun = act)(d)
	model = models.Model(inputs=inputs, outputs=x)
	if preT:
		print("preT ae ae")
		model.set_weights(init_weights)
	return(model)

def linked_as_dt_old(encoding_dim, gene_num, act, init,seed, kl1, kl2, preT=False, init_weights=[]):
	inputs = Input(shape=(gene_num,))
	d = Dropout(0.1)(inputs)
	dense_1 = keras.layers.Dense(100, activation="selu")
	dense_2 = keras.layers.Dense(30, activation="selu")
	tied_encoder = keras.models.Sequential([
		inputs,
		dense_1,
		dense_2
		])

	tied_decoder = keras.models.Sequential([
		DenseTranspose(dense_2, activation="selu"),
		DenseTranspose(dense_1, activation="sigmoid")
		])
	tied_ae = keras.models.Sequential([tied_encoder, tied_decoder])
	return(tied_ae)

def linked_as_dt(encoding_dim, gene_num, act, init,seed, kl1, kl2, preT=False, init_weights=[]):
	inputs = Input(shape=(gene_num,))
	d = Dropout(0.1)(inputs)
	dense_1 = keras.layers.Dense(100, activation="selu")
	dense_2 = keras.layers.Dense(30, activation="selu")
	tied_encoder = keras.models.Sequential()
	tied_encoder.add(inputs)
	tied_encoder.add(dense_1)
	tied_encoder.add(dense_2)

	tied_encoder.add(DenseTranspose(dense_2, activation="selu"))
	tied_encoder.add(DenseTranspose(dense_1, activation="selu"))

	#tied_ae = keras.models.Sequential([tied_encoder, tied_decoder])
	return(tied_encoder)
