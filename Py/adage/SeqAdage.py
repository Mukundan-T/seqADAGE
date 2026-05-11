"""
Georgia Doing 2025
SeqADAGE class

A class for an RNA-seq based ADAGE model. Inherits ADAGE class.

Usage
    constructor
    SeqADAGE.train_model()

"""

from adage import Adage as ad
from adage import LinkedAE as lae
import pandas as pd
import numpy as np
from tensorflow.keras import optimizers, losses


class SeqAdage(ad.Adage): 
    """
    Adage instance with associated linked denoising autoencoder (LinkedAE), 
    training data, and hyperparameter settings. Weight matrix used to 
    assign Adage atributes, namely high weight genes and gene set 
    enrichment results.

    Attributes:
        autoencoder (LinkedAE): The tied-weights denoising autoencoder
        all_comp (pd.df): Pandas DataFrame of training data
        gene_num (int): Number of genes (deatures, dims) in input data
        enc_dim (int): Encoding dimension, number of nodes in hidden layer
        epochs (int): Number of epochs for training
        seed (int): Random seed
        batch_size (int): Number of samples per training batch
        mm (flt): Momentum for fitting, 0-1
        lr (flt): Learning rate, 0-1
        v (int): Verbose, 1-True, 0-False

    Methods:
        prep_data() : prepares data by introducing noise
        train_model(): Fits autoencoder to training data

    To Do:
        * check autoencoder weights update after training/fitting
        * reconcile with Adage constructor, super().__init__()
        * prep E. coli KEGG, GO, operon and regulon gene sets
        * implement hyperparam search
        * implement model saving 
    """
    
    def __init__(self, input_file, seed=100, enc_dim=10 ,kl1=0, kl2=0, 
                 act="tanh", act2="tanh", tied=True, epochs=50, 
                 init="glorot_uniform", batch_size=10, dropout=0, mm=0, 
                 lr=0.01, v=1):
        """Constructor for SeqAdage with hyperparameters

        This class inherits the Adage class, instantiating a linked
        denoising autoencoder with a set of hyperparameter values, fitting
        it to a training dataset and saving weight and bias matrices to files.

        Args:
            input_file (str): Name of csv with training data
            seed (int): Random seed
            enc_dim (int): Encoding dimension, number of nodes in hidden layer
            kl1 (flt): Amount of L1 regularizatotion, 0-1
            kl2 (flt): Amount of L2 regularizatotion, 0-1
            act (str): Activation function during encoding
            act2 (str): Activation function during decoding
            tied (bool): Whether ncoder and decoder weights are tied
            epochs (int): Number of epochs for training
            init (str): Inititalization function for random weight distribution
            batch_size (int): Number of samples per training batch
            dropout (flt): Proportion of input values randomly dropped
            mm (flt): Momentum for fitting, 0-1
            lr (flt): Learning rate, 0-1
            v (int): Verbose, 1-True, 0-False
        """
        self.autoencoder = lae.LinkedAE(enc_dim, act2, dropout)
        self.all_comp = pd.read_csv(input_file, index_col=0)
        self.gene_num = np.size(self.all_comp, 0)
        self.encoding_dim = enc_dim
        self.epochs = epochs
        self.seed = seed
        self.batch_size = batch_size
        self.lr = lr
        self.mm = mm
        self.v = v
        self.history = None

    def prep_data(self):
        """Prepares training data by adding noise. Called by train_model()"""
        all_comp_np = self.all_comp.values.astype("float64")
        gene_num = np.size(all_comp_np, 0)
        x_train = all_comp_np.transpose()
        # set noise factor
        noise_factor = 0.1
        x_train_noisy = x_train + (noise_factor* np.random.normal(loc=0.0,scale=1.0, size=x_train.shape))
        # limit to range 0-1
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)

        return(x_train, x_train_noisy)

    def train_model(self):
        """Fits the autoencoder model to the training set"""
        np.random.seed(self.seed)
        x_train, x_train_noisy = self.prep_data()
        optim = optimizers.SGD(learning_rate=self.lr, momentum=self.mm) # lr=0.001, rho=0.95, epsilon=1e-07
        self.autoencoder.compile(optimizer=optim, 
                        loss=losses.MeanSquaredError()) #BinaryCrossentropy(from_logits=False)) 

        history = self.autoencoder.fit(x=x_train, 
                                  y=x_train_noisy, 
                                  epochs=self.epochs,
                                  batch_size=self.batch_size,
                                  #shuffle=True,
                                  validation_split = 0.1,
                                  verbose=self.v
                                 )
        print("fitted")
        #self.autoencoder = autoencoder
        self.history = history
        return(self.autoencoder)
        
