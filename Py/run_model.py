"""

Georgia Doing 2020
Denoising Autoencoder with P.a. RNAseq

Usage:

	run_model.py <dataset> [--seed=<SEED>]

Options:
	-h --help			Show this screen
	<dataset>			File patht to RNAseq compnedium
	--seed = <SEED>		Random seed for training [default: 1]

Output:

	**Weights, Loss and validation loss saved as files
"""


import tensorflow as tf
from tensorflow import keras

import argparse
import numpy as np
import csv
import pandas as pd

from tensorflow.keras import optimizers, regularizers, layers, initializers, models
from tensorflow.keras.layers import Input, Dense, Dropout

#import TiedWeightsEncoder as tw
import Adage as ad
#from AdageHyperModel import AdageHyperModel
from AdageHyperModelv2 import AdageHyperModelv2

#from autoencoders import DenseLayerAutoencoder

from DenseTranspose import DenseTranspose

from keras.models import Model, load_model
import keras_tuner
import matplotlib.pyplot as plt
import os







def tune_model(input_file, seed):
    """
	
	"""
    hp = keras_tuner.HyperParameters()
	# defining the number of neurons dynamically
    units = hp.Int(name="units", min_value=10, max_value=100, step=10)   
    # defining the dropout rate
    dropout = hp.Float(name="dropout", min_value=0.0, max_value=0.9, step = 0.1)
    # Automatically assign True/False values.
    act1 = hp.Choice('act1', ["sigmoid","tanh","relu", "celu"])
    act2 = hp.Choice('act2', ["sigmoid","tanh","relu", "celu"])
    shuffle = hp.Boolean("shuffle", default=False)
    init = hp.Choice('init', ["glorot_uniform","glorot_normal"]) 
    kl1 = hp.Float('kl1', min_value = 0, max_value = .15, step = 0.005)
    kl2 = hp.Float('kl2', min_value = 0, max_value = .9, step = 0.05)
    #al1 = hp.Float('al2', min_value = 0, max_value = 1, step = 0.1)
    lr = hp.Float('lr', min_value = 0.001, max_value = 0.1, step = 0.01) 
    bs = hp.Int('bs', min_value = 5, max_value = 100, step = 5) 
    mm = hp.Float('mm', min_value = 0.0, max_value = .9, step = 0.1)

    all_comp = pd.read_csv(input_file, index_col=0)
    gene_num = np.size(all_comp, 0)

    tuner = keras_tuner.Hyperband(
                       hypermodel=AdageHyperModelv2(input_shape=gene_num),#
		               hyperparameters = hp,
                       objective = "val_loss", #optimize val acc
                       max_epochs=50, #for each candidate model
                       overwrite=True,  #overwrite previous results
                       directory='/work/gd134/hyperband_search_dir', #Saving dir
                       project_name=input_file.removesuffix(".csv").replace("../data_files/","")+"_" + str(seed))
    
    
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
    
def test_model(input_file, post_data = '', seed=123, enc_dim = 300, epochs=50, kl1=0, kl2=0, lr = 0.01,
			  act='sigmoid', init='glorot_uniform', tied=True, batch_size=10,
			  v=1, pre_w = ''):
	"""

	"""
	all_comp = pd.read_csv(input_file, index_col=0)
	
	# this is the size of our input
	gene_num = np.size(all_comp, 0)
	# size of hidden layer
	encoding_dim = enc_dim

	# prepare data
	x_train, x_train_noisy = prep_data(all_comp, seed)
	autoencoder = linked_as_dt(encoding_dim, gene_num, act, init,
								seed, kl1, kl2)


	autoencoder.save('dense_autoencoder_test.keras')

	del autoencoder

	autoencoder = load_model('dense_autoencoder_test.keras', 
							 custom_objects={'DenseLayerAutoencoder': DenseLayerAutoencoder})
	print(autoencoder.summary())
	config = keras.layers.serialize(autoencoder)
	print(config)
	return(autoencoder)

def run_model(input_file, post_data = '', seed=123, enc_dim = 300, epochs=50, kl1=0, kl2=0, lr = 0.01,
			  act='sigmoid', act2='sigmoid', init='glorot_uniform', tied=True, batch_size=10,dropout=0, mm=0,
			  v=1, pre_w = ''):
	"""

	"""
	## check if there is data from post fine-tuning
	if(post_data == ''):
		post_data = input_file
	## read in training data
	all_comp = pd.read_csv(input_file, index_col=0)
	## this is the size of our input
	gene_num = np.size(all_comp, 0)
	#  this is the size of hidden layer
	encoding_dim = enc_dim
	# prepare data
	x_train, x_train_noisy = prep_data(all_comp, seed)

	if(tied):
		print("tied")
		autoencoder = linked_ae_dt(encoding_dim, gene_num, act,act2, init,
								seed, kl1, kl2)
	else:
		autoencoder = unlinked_ae_dt(encoding_dim, gene_num, act,act2, init,
								seed, kl1, kl2)

	autoencoder, history = train_model(autoencoder, x_train,
		                               x_train_noisy, epochs,
									   seed, batch_size, lr, v,mm)

    # save second model before fine-tuning
	weights_tmp, b_weights_tmp = autoencoder.get_weights()[0:2]
	file_desc = ( input_file[13:-4] + '_'
	             #+ post_data[14:-4]
	             + '_s' + str(seed)
				 + '_n' + str(encoding_dim)
				 + "_k1" + str(kl1)
				 + "_k2" + str(kl2)
				 +  "_a1" + act
				 +  "_a2" + act2
				 + '_i' + init
				 + '_e' + str(epochs)
				 + '_t' + str(tied)
				 + '_b' + str(batch_size)
				 + '_l' + str(lr)
				 + '_m' + str(mm)
				 + '_d' + str(dropout)
				 + '_p')
	
	write_data(file_desc + '_pRM', weights_tmp, b_weights_tmp, history)




	adage = ad.Adage(autoencoder, history, all_comp)

	return adage



def linked_ae_dt(encoding_dim, gene_num, act,act2, init,seed, kl1, kl2,dropout=0, preT=False, init_weights=[]):
	inputs = Input(shape=(gene_num,))
	d = Dropout(dropout)
	dense_1 = keras.layers.Dense(encoding_dim, activation=act,
		kernel_regularizer = regularizers.l1_l2(l1=kl1, l2=kl2),
		#activity_regularizer = regularizers.l1(0),
		kernel_initializer = init, name = "dense_1")
	#dense_2 = keras.layers.Dense(30, activation="selu")
	#if preT:
	#	dense_1.set_weights(init_weights)	
	tied_encoder = keras.models.Sequential()
	tied_encoder.add(inputs)
	tied_encoder.add(d)
	tied_encoder.add(dense_1)
	#tied_encoder.add(dense_2)

	#tied_encoder.add(DenseTranspose(dense_2, activation="selu"))
	tied_encoder.add(DenseTranspose(dense_1, activation=act2))

	#tied_ae = keras.models.Sequential([tied_encoder, tied_decoder])
	return(tied_encoder)


def unlinked_ae_dt(encoding_dim, gene_num, act,act2, init,seed, kl1, kl2,dropout=0, preT=False, init_weights=[]):
	inputs = Input(shape=(gene_num,))
	d = Dropout(dropout)
	dense_1 = keras.layers.Dense(encoding_dim, activation=act,
		kernel_regularizer = regularizers.l1_l2(l1=kl1, l2=kl2),
		#activity_regularizer = regularizers.l1(0),
		kernel_initializer = init, name = "dense_1")

	dense_2 = keras.layers.Dense(gene_num, activation=act2,
		kernel_regularizer = regularizers.l1_l2(l1=kl1, l2=kl2),
		#activity_regularizer = regularizers.l1(0),
		kernel_initializer = init, name = "dense_2")
	
	#dense_2 = keras.layers.Dense(30, activation="selu")
	#if preT:
	#	dense_1.set_weights(init_weights)	
	tied_encoder = keras.models.Sequential()
	tied_encoder.add(inputs)
	tied_encoder.add(d)
	tied_encoder.add(dense_1)
	tied_encoder.add(dense_2)

	#tied_encoder.add(DenseTranspose(dense_2, activation="selu"))
	#tied_encoder.add(DenseTranspose(dense_1, activation=act))

	#tied_ae = keras.models.Sequential([tied_encoder, tied_decoder])
	return(tied_encoder)


	

def prep_data(all_comp, seed):
	all_comp_np = all_comp.values.astype("float64")
	gene_num = np.size(all_comp_np, 0)
	x_train = all_comp_np.transpose()
	# set noise factor
	noise_factor = 0.1
	# add noise
	x_train_noisy = x_train + (noise_factor
		            * np.random.normal(loc=0.0,scale=1.0, size=x_train.shape))
	# limit to range 0-1
	x_train_noisy = np.clip(x_train_noisy, 0., 1.)

	return(x_train, x_train_noisy)

def transf_data(weights):
	weights_copy = []
	print(weights)
	print(len(weights))
	for w in weights:
		print(w.shape)
		weights_copy.append(np.zeros((w.shape)))
	print(weights_copy)


def train_model(autoencoder, x_train, x_train_noisy, epochs, seed, batch_size, lr, v, mm=0):

	np.random.seed(seed)
	train_idxs = np.random.choice(x_train.shape[0],
							      int(x_train.shape[0]*0.9), replace=False)
	x_train_train = x_train[train_idxs,:]
	x_train_test = x_train[~np.in1d(range(x_train.shape[0]),train_idxs),:]

	x_train_noise_train = x_train_noisy[train_idxs,:]
	x_train_noise_test = x_train_noisy[~np.in1d(range(x_train.shape[0]),
		                                              train_idxs),:]

	optim = optimizers.SGD(learning_rate=lr, momentum=mm) # lr=0.001, rho=0.95, epsilon=1e-07
	autoencoder.compile(optimizer=optim, 
                        loss=tf.keras.losses.MeanSquaredError()) #BinaryCrossentropy(from_logits=False)) 
	x_train_f = np.clip(x_train, 0., 1.).astype("float32") 
	x_train_noisy_f = np.clip(x_train_noisy, 0., 1.).astype("float32") 

	history = autoencoder.fit(x=x_train, #_noisy_f
                              y=x_train_noisy, #_f
                              epochs=epochs,
                              batch_size=batch_size,
                              #shuffle=True,
                              validation_split = 0.1,
                              #verbose=1,
                              #validation_data=(np.array(x_train_noise_test),
                              #				 np.array(x_train_test)),
                              verbose=v
                             )
	print("fitted")
	return(autoencoder, history)


def write_data(file_desc, weights, b_weights, history):
	"""
	Save logs and output for a model in an outputs foolder
	"""
	np.savetxt('../outputs/weights/' + file_desc + '_ew_da.csv',
		np.matrix(weights), fmt = '%s', delimiter=',')
	np.savetxt('../outputs/bias/' + file_desc + '_eb_da.csv',
		np.matrix(b_weights), fmt = '%s', delimiter=',')
	np.savetxt('../outputs/loss/' + file_desc + '_l_da.csv',
		np.matrix(history.history['loss']), fmt = '%s', delimiter=',')
	np.savetxt('../outputs/val_loss/' + file_desc + '_vl_da.csv',
		np.matrix(history.history['val_loss']), fmt = '%s', delimiter=',')


if __name__ == '__main__':
		parser = argparse.ArgumentParser(description='Get training set.')
		parser.add_argument('filename',type=str, nargs=1,
			help='filpath to training set.')
		args=parser.parse_args()
		run_model(args.filename[0])
