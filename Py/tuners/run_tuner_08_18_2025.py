import run_model_preT
import random
import numpy as np
#import run_model
#import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#import numpy as np
#from imp import reload
#import Adage
#from scipy.stats import hypergeom
#import csv
#from AdageHyperModelv2 import AdageHyperModelv2
#import tensorflow as tf
#import shap
#from keras.models import Model, load_model
#import keras_tuner as kt

np.random.seed(100)

for s in random.sample(range(1000),k=5):
	run_model_preT.tune_model('../data_files/ecoli_MG165_genome_part2_log_counts_norm_01.csv',s)
	#run_model_preT.tune_model('../data_files/saur_pan_genome_log_counts_norm_01.csv',s)
	#run_model_preT.tune_model('../data_files/sepi_pan_genome_log_counts_norm_01.csv',s)
	#run_model_preT.tune_model('../data_files/saur_pan_genome_BLAST_log_counts_norm_01.csv',s)
	#run_model_preT.tune_model('../data_files/saur_pan_genome_BLAST_with_rand_log_counts_norm_01.csv',s)


for s in range(5):
	run_model_preT.tune_model('../data_files/ecoli_MG165_genome_part2_log_counts_norm_01.csv',960+s)
	#run_model_preT.tune_model('../data_files/saur_pan_genome_log_counts_norm_01.csv',960+s)
	#run_model_preT.tune_model('../data_files/sepi_pan_genome_log_counts_norm_01.csv',960+s)
	#run_model_preT.tune_model('../data_files/saur_pan_genome_BLAST_log_counts_norm_01.csv',960+s)
	#run_model_preT.tune_model('../data_files/saur_pan_genome_BLAST_with_rand_log_counts_norm_01.csv',960+s)

