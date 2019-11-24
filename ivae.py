seed_value= 0
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.set_random_seed(seed_value)
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth = True 
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
tf.keras.backend.set_session(sess)

from time import time
from dataset import load_one2one_interaction_mnist, load_interaction_mnist
from keras.optimizers import Adam
from keras.layers import Lambda, Input, Dense, Average
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy, kullback_leibler_divergence
from keras.utils import plot_model
from keras import backend as K
from keras.layers import Concatenate
from keras.callbacks import ModelCheckpoint
import keras

import numpy as np
my_seed = 1024
np.random.seed(my_seed)
import matplotlib.pyplot as plt
import argparse
import os
from tensorflow.python.keras.callbacks import TensorBoard


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class IVAE_checkpointer(keras.callbacks.Callback):
    def init_model(self, models, model_paths):
        self.model_list = models
        self.path_list = model_paths
        self.temp_loss = 999999
        
    def on_epoch_end(self, epoch, logs={}):
        val_loss = float(logs['val_loss'])
        if self.temp_loss > val_loss:
            self.temp_loss = val_loss
            for m, p in zip(self.model_list, self.path_list):
                m.save(p)
                print("model saved {} ".format(p))
    

class Interactional_VAE():
    def __init__(self, f_dims=784, n_f=2, h_dims=[512,20], l_mode="all", tag="ivae", save_path="models"):
        self.mode_loss = l_mode # all, separated only, concatenated only
        self.f_dims = f_dims
        self.n_features = n_f
        self.i_shape = (f_dims,)
        self.hidden_dims = h_dims[0]
        self.latent_dims = h_dims[1]
        self.tag = tag
        self.tensorboard = TensorBoard(log_dir="{}/{}/logs/{}".format(save_path,tag,time()))
        
        self.save_path = "{}/{}".format(save_path, tag)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        
        isSummary = True
        self._init_encoders(isSummary=isSummary)
        self._init_reducer(isSummary=isSummary)
        self._init_decoders(isSummary=isSummary)    
        self._concat_models(isSummary=isSummary)
        self._init_loss()
        
        
    def _init_encoders(self, isSummary=True, isSaveFig=True):
        """ 
        ENCODER
        : 각 입력 데이터를 정규분포 모델로 변환하는 역할을 수행
        """
        
        self.inputs     = [Input(shape=self.i_shape, name="{}_enc_input_{}".format(self.tag, i+1)) 
                           for i in range(self.n_features)]
        self.e_hiddens  = [Dense(self.hidden_dims, activation="relu", name="{}_enc_hidden_{}".format(self.tag, i+1))(self.inputs[i]) 
                           for i in range(self.n_features)]
        self.z_means    = [Dense(self.latent_dims, name="{}_enc_z_mean_{}".format(self.tag, i+1))(self.e_hiddens[i]) 
                           for i in range(self.n_features)]
        self.z_log_vars = [Dense(self.latent_dims, name="{}_enc_z_log_var_{}".format(self.tag, i+1))(self.e_hiddens[i]) 
                           for i in range(self.n_features)]
        self.encoders   = [Model(self.inputs[i], [self.z_means[i], self.z_log_vars[i]], name="{}_enc_{}".format(self.tag, i+1)) 
                           for i in range(self.n_features)]
        
        self.encoder_save_paths = ["{}/encoder_{}.h5".format(self.save_path, i+1) for i in range(self.n_features)]
        
        if isSummary:
            print("")
            print("_"*100)
            print("ENCODER")
            for i in range(self.n_features):
                self.encoders[i].summary()
                
        if isSaveFig:
            for i in range(self.n_features):
                plot_model(self.encoders[i], to_file="imgs/{}_encoder_{}.png".format(self.tag, i+1), show_shapes=True)
    
    def _init_reducer(self, isSummary=True, isSaveFig=True):
        """ 
        REDUCER
        : 각 입력 데이터의 정규분포 모델의 평균과 분산 값을 입력으로 받아 모으는 역할 수행
        """
        self.latent_means    = [Input(shape=(self.latent_dims,), name="{}_red_mean_{}".format(self.tag, i+1)) 
                                for i in range(self.n_features)]
        self.latent_log_vars = [Input(shape=(self.latent_dims,), name="{}_red_log_var_{}".format(self.tag, i+1)) 
                                for i in range(self.n_features)]
        
        self.latent_mean    = Average()(self.latent_means)
        self.latent_log_var = Average()(self.latent_log_vars)
               
        self.latent         = Lambda(sampling, output_shape=(self.latent_dims * self.n_features,), 
                                     name="{}_latent".format(self.tag))([self.latent_mean, self.latent_log_var])
        self.reducer        = Model(self.latent_means + self.latent_log_vars, [self.latent, self.latent_mean, self.latent_log_var])
        self.reducer_save_path = "{}/reducer.h5".format(self.save_path)
        
        
        if isSummary:
            print("")
            print("_"*100)
            print("REDUCER")
            self.reducer.summary()
        
        if isSaveFig:
            plot_model(self.reducer, to_file="imgs/{}_reducer.png".format(self.tag), show_shapes=True)
        
        
    def _init_decoders(self, isSummary=True, isSaveFig=True):
        """ 
        DECODERs
        : 통합된 정규분포 모델의 평균과 분산 값을 통해 샘플링을 통한 VAE decoders
        """
        self.latent_inputs = Input(shape=(self.latent_dims, ), name="{}_dec_latent_inputs".format(self.tag))
        self.d_hiddens     = [Dense(self.hidden_dims, activation="relu", name="{}_dec_hidden_{}".format(self.tag, i+1))(self.latent_inputs) 
                              for i in range(self.n_features)]
        self.outputs       = [Dense(self.f_dims, activation="sigmoid", name="{}_dec_output_{}".format(self.tag, i+1))(self.d_hiddens[i]) 
                              for i in range(self.n_features)]
        self.decoder       = Model(self.latent_inputs, self.outputs, name="{}_decoder".format(self.tag))
        self.decoder_save_path = "{}/decoder.h5".format(self.save_path)
        
        if isSummary:
            print("")
            print("_"*100)
            print("DECODER")
            self.decoder.summary()
            
        if isSaveFig:
            plot_model(self.decoder, to_file="imgs/{}_decoder.png".format(self.tag), show_shapes=True)
                        
    
    def _concat_models(self, isSummary=True, isSaveFig=True):
        self.concat_outputs = self.decoder(self.reducer(self.latent_means + self.latent_log_vars)[0])
        self.concat_decoder = Model(self.latent_means + self.latent_log_vars, self.concat_outputs, name="{}_con_dec".format(self.tag))
        self.concat_decoder_save_path = "{}/latent_decoder.h5".format(self.save_path)
        
        if isSummary:
            print("")
            print("_"*100)
            print("CONCATENATED-DECODER")
            self.concat_decoder.summary()
        
        self.ivae_z_means, self.ivae_z_log_vars = [], []
        
        for i in range(self.n_features):
            _z_mean, _z_log_var = self.encoders[i](self.inputs[i])
            self.ivae_z_means += [_z_mean]
            self.ivae_z_log_vars += [_z_log_var]
        self.ivae_z = self.ivae_z_means + self.ivae_z_log_vars
        """
        for i_layer, encoder in zip(self.inputs, self.encoders):
            _z_mean, _z_log_var = encoder(i_layer)
            self.ivae_z_means += [_z_mean]
            self.ivae_z_log_vars += [_z_log_var]
        self.ivae_z = self.ivae_z_means + self.ivae_z_log_vars
        """
        
        self.ivae_latents        = self.reducer(self.ivae_z)
        self.ivae_latent         = self.ivae_latents[0]
        self.ivae_latent_mean    = self.ivae_latents[1]
        self.ivae_latent_log_var = self.ivae_latents[2]
        self.ivae_encoder        = Model(self.inputs, self.ivae_latent)
        self.ivae_outputs        = self.decoder(self.ivae_encoder(self.inputs))
        self.ivae                = Model(self.inputs, self.ivae_outputs, name="ivae")
        self.ivae_save_path = "{}/training_ivae.h5".format(self.save_path)
        
        if isSummary:
            print("")
            print("_"*100)
            print("INTERACTIONAL-VAE")
            self.ivae.summary()
        
    
    def _init_loss(self, scale_reducer=1.0, scale_others=1.0):
        # 1. Reconstruction Error of Each Decoders
        self.rec_losses = [binary_crossentropy(self.inputs[i], self.ivae_outputs[i]) * (self.f_dims) 
                           for i in range(self.n_features)]
        
        # 2. KL-loss of Each Encoders
        self.kl_losses = []
        for z_mean, z_log_var in zip(self.ivae_z_means, self.ivae_z_log_vars):
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            self.kl_losses.append(kl_loss)
        
        # 3. KL-loss of the Reducer
        red_kl_loss = 1 + self.ivae_latent_log_var - K.square(self.ivae_latent_mean) - K.exp(self.ivae_latent_log_var)
        red_kl_loss = K.sum(red_kl_loss, axis=-1)
        red_kl_loss *= -0.5
        red_kl_loss = K.mean(red_kl_loss)
        
        
        # 4. Equivalence Space mapping regularizer
        self.eq_losses = []
        for i, (z_mean_a, z_log_var_a) in enumerate(zip(self.ivae_z_means, self.ivae_z_log_vars)):
            enc_kl_losses = []
            for j, (z_mean_b, z_log_var_b) in enumerate(zip(self.ivae_z_means, self.ivae_z_log_vars)):
                if i != j:
                    enc_kl_loss = K.exp(K.square(z_mean_a - z_mean_b) + z_log_var_a + z_log_var_b)
                    enc_kl_loss += (z_log_var_a - z_log_var_b - 1)
                    enc_kl_loss = K.sum(enc_kl_loss, axis=-1)
                    enc_kl_losses.append(enc_kl_loss)
            self.eq_losses.append(enc_kl_losses)
        
        
        # VAE loss of Each (Encoder + Decoder) + Reducer's error
        self.vae_losses = []
        for rec_loss, kl_loss, enc_kl_losses in zip(self.rec_losses, self.kl_losses, self.eq_losses):
            vae_loss = rec_loss + kl_loss
            
            for enc_kl_loss in enc_kl_losses:
                vae_loss += enc_kl_loss
                
            vae_loss = K.mean(vae_loss)
            self.vae_losses.append(vae_loss)
                
        
        # Concat all losses
        self.vae_loss = (red_kl_loss * scale_reducer)
        for loss in self.vae_losses:
            self.vae_loss += (loss * scale_others)
        
        self.ivae.add_loss(self.vae_loss)
        
        """
        for rec_loss, kl_loss, enc_kl_losses in zip(self.rec_losses, self.kl_losses, self.eq_losses):
            vae_loss = rec_loss + kl_loss
            
            for enc_kl_loss in enc_kl_losses:
                vae_loss += enc_kl_loss
                
            vae_loss = K.mean(vae_loss)
            self.ivae.add_loss(vae_loss)
        
        self.ivae.add_loss(red_kl_loss)
        """
    
    def compile(self, lr=1e-4, optimizer=Adam):
        self.ivae.compile(optimizer=optimizer(lr))
        print("IVAE compile finished!")
        
    
    def fit(self, epochs, batch_size, train_data, test_data, model_save_path, model_save_name):
        train_x1, train_x2 = train_data
        test_x1, test_x2 = test_data
        
        train_ckpt = IVAE_checkpointer() 
        save_models = self.encoders + [self.reducer, 
                                       self.decoder,
                                       self.concat_decoder,
                                       self.ivae]
        
        save_paths  = self.encoder_save_paths + [self.reducer_save_path, 
                                                 self.decoder_save_path,
                                                 self.concat_decoder_save_path, 
                                                 self.ivae_save_path]
        
        train_ckpt.init_model(models=save_models, model_paths=save_paths)
        
        self.ivae.fit([train_x1, train_x2],
                      epochs=epochs,
                      batch_size=batch_size,
                      callbacks=[train_ckpt, self.tensorboard],
                      shuffle=True,
                      validation_data=([test_x1, test_x2], None))
      

    
    
    
def experiment_MNIST():
    # Load MNIST 
    (x_train, x1_train, x2_train),(x_test, x1_test, x2_test) = load_interaction_mnist() 
    train_data = (x1_train, x2_train)
    test_data = (x1_test, x2_test)
        
    # MNIST experiment
    model = Interactional_VAE(f_dims=784, n_f=2, h_dims=[512,50], tag="ivae_v1_mnist")
    model.compile(lr=1e-4, optimizer=Adam)
    model.fit(epochs=300,
              batch_size=1024, 
              train_data=train_data,
              test_data=test_data,
              model_save_path="models", 
              model_save_name="ivae_v1_mnist.h5")
    

def experiment_one2one_relational_MNIST():
    # Load MNIST 
    (x_train, x1_train, x2_train),(x_test, x1_test, x2_test) = load_one2one_interaction_mnist() 
    train_data = (x1_train, x2_train)
    test_data = (x1_test, x2_test)
        
    # MNIST experiment
    model = Interactional_VAE(f_dims=784, n_f=2, h_dims=[512,50], tag="ivae_v1_o2o_mnist")
    model.compile(lr=1e-4, optimizer=Adam)
    model.fit(epochs=150,
              batch_size=1024, 
              train_data=train_data,
              test_data=test_data,
              model_save_path="models", 
              model_save_name="ivae_v1_o2o_mnist.h5")

    
if __name__ == "__main__":
    #experiment_MNIST()
    experiment_one2one_relational_MNIST()
    
    
    