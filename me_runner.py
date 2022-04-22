##################################################################################
# VAE on DarkMachine dataset with 3D Sparse Loss                                 # 
# Author: B. Orzani (Universidade Estadual Paulista, Brazil), M. Pierini (CERN)  #
##################################################################################

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from pickle import dump
import numpy as np
import h5py
from tqdm import tqdm

from data_utils import save_npy, save_csv, read_npy, save_run_history
from network_utils import train_convnet, test_convnet
import VAE_NF_Conv2D as VAE


class ConvNetRunner:
    def __init__(self, args):

        # Hyperparameters
        self.data_save_path = args.data_save_path
        self.model_save_path = args.model_save_path
        self.Data_filename = args.Data_filename
        self.Data_bsm_filename = args.Data_bsm_filename
        self.Met_filename = args.Met_filename
        self.Met_bsm_filename = args.Met_bsm_filename
        self.model_name = args.model_name
        self.num_epochs = args.num_epochs
        self.num_classes = args.num_classes
        self.training_fraction = args.training_fraction
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.learning_rate = args.learning_rate
        self.latent_dim = args.latent_dim
        self.beta = args.beta
        # self.test_model_path = args.test_model_path
        self.test_data_save_path = args.test_data_save_path

        self.network = args.network
        self.flow = args.flow 
        # print(args.flow, self.flow)

        if self.flow == 'noflow':
            self.model = VAE.ConvNet(args)
            self.flow_ID = 'NoF'
        elif self.flow == 'planar':
            self.model = VAE.PlanarVAE(args)
            self.flow_ID = 'Planar'
        elif self.flow == 'orthosnf':
            self.model = VAE.OrthogonalSylvesterVAE(args)
            self.flow_ID = 'Ortho'
        elif self.flow == 'householdersnf':
            self.model = VAE.HouseholderSylvesterVAE(args)
            self.flow_ID = 'House'
        elif self.flow == 'triangularsnf':
            self.model = VAE.TriangularSylvesterVAE(args)
            self.flow_ID = 'Tri'
        elif self.flow == 'iaf':
            self.model = VAE.IAFVAE(args)
            self.flow_ID = 'IAF'
        elif self.flow == 'convflow':
            self.model = VAE.ConvFlowVAE(args)
            self.flow_ID = 'ConvF'
        else:
            raise ValueError('Invalid flow choice')
        
        self.model_name = self.model_name%self.flow_ID
        self.model = self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.preprocess_data()

    def preprocess_data(self):

        outerdata_train = np.load("/global/u2/a/agarabag/anomoly_studies/CATHODE/separated_data/outerdata_train.npy")
        outerdata_test = np.load("/global/u2/a/agarabag/anomoly_studies/CATHODE/separated_data/outerdata_test.npy")

        outerdata_train = outerdata_train[outerdata_train[:,5]==0]
        outerdata_test = outerdata_test[outerdata_test[:,5]==0]

        nFeat = 4

        data_train = outerdata_train[:,1:nFeat+1]
        print('shape of data_train: ', data_train.shape)
        data_test = outerdata_test[:,1:nFeat+1]
        print('shape of data_test: ', data_test.shape)

        data = np.concatenate((data_train, data_test), axis=0)
        print('shape of data: ', data.shape)
        
        
        trainsize = 500000
        print(np.shape(data))
        x_train = data[:trainsize]
        x_test = data[trainsize:]
        y_train = cond_data[:trainsize]
        y_test = cond_data[trainsize:]

        # x_train = np.hstack([x_train,y_train.reshape(y_train.shape[0],1)])
        # x_test = np.hstack([x_test,y_test.reshape(y_test.shape[0],1)])

        image_size = x_train.shape[1]
        original_dim = image_size
        x_train = np.reshape(x_train, [-1, original_dim])
        x_test = np.reshape(x_test, [-1, original_dim])
        self.x_train = x_train.astype('float32')
        self.x_test = x_test.astype('float32')
        y_train = np.reshape(y_train, [-1, 1])
        y_test = np.reshape(y_test, [-1, 1])
        self.y_train = y_train.astype('float32')
        self.y_test = y_test.astype('float32')
        
        self.batch_size = 1024


#         # build train and val sets
#         i_train = int(self.d.shape[0]*self.training_fraction)
#         # training data
#         self.x_train = self.d[:i_train,:,:,:]
#         self.met_train = self.Met_d[:i_train,:] 
#         self.weight_train = self.weight[:i_train]
        
#         # Val data
#         self.x_val = self.d[i_train:,:,:,:]
#         self.met_val = self.Met_d[i_train:,:]
#         self.weight_val = self.weight[i_train:]
        
        

    def trainer(self):
        self.train_loader = DataLoader(dataset = self.x_train, batch_size = self.batch_size, shuffle=True)
        self.metTr_loader = DataLoader(dataset = self.y_train, batch_size = self.batch_size, shuffle=True)

        self.val_loader = DataLoader(dataset = self.x_test, batch_size = self.batch_size, shuffle=False)
        self.metVa_loader = DataLoader(dataset = self.y_test, batch_size = self.batch_size, shuffle=False)

        # to store training history
        self.x_graph = []
        self.train_y_rec = []
        self.train_y_kl = []
        self.train_y_loss = []
        self.val_y_rec = []
        self.val_y_kl = []
        self.val_y_loss = []

        # print('Model Parameter: ', self.model)
        print('Model Type: %s'%self.flow_ID)
        print('Initiating training, validation processes ...')
        for epoch in range(self.num_epochs):
            self.x_graph.append(epoch)
            print('Starting to train ...')

            # training
            tr_loss_aux = 0.0
            tr_kl_aux = 0.0
            tr_rec_aux = 0.0
            for y, (x_train, met_tr, wt_train) in tqdm(enumerate(zip(self.train_loader, self.metTr_loader, self.weight_train_loader))):
                if y == (len(self.train_loader)): break

                tr_loss, tr_kl, tr_eucl, self.model = train_convnet(self.model, x_train, met_tr, wt_train, self.optimizer, batch_size=self.batch_size)
                
                tr_loss_aux += tr_loss
                tr_kl_aux += tr_kl
                tr_rec_aux += tr_eucl

            print('Moving to validation stage ...')
            # validation
            val_loss_aux = 0.0
            val_kl_aux = 0.0
            val_rec_aux = 0.0

            for y, (x_val, met_va, wt_val) in tqdm(enumerate(zip(self.val_loader, self.metVa_loader, self.weight_val_loader))):
                if y == (len(self.val_loader)): break
                
                #Test
                val_loss, val_kl, val_eucl = test_convnet(self.model, x_val, met_va, wt_val, batch_size=self.batch_size)

                val_loss_aux += val_loss
                val_kl_aux += val_kl
                val_rec_aux += val_eucl

            self.train_y_loss.append(tr_loss_aux.cpu().detach().numpy()/(len(self.train_loader)))
            self.train_y_kl.append(tr_kl_aux.cpu().detach().numpy()/(len(self.train_loader)))
            self.train_y_rec.append(tr_rec_aux.cpu().detach().numpy()/(len(self.train_loader)))
                
            self.val_y_loss.append(val_loss_aux/(len(self.val_loader)))
            self.val_y_kl.append(val_kl_aux/(len(self.val_loader)))
            self.val_y_rec.append(val_rec_aux/(len(self.val_loader)))
                
            print('Epoch: {} -- Train loss: {}  -- Val loss: {}'.format(epoch, 
                                                                         tr_loss_aux/(len(self.train_loader)), 
                                                                         val_loss_aux/(len(self.val_loader))))
            if (epoch == 0):
                self.best_val_loss = val_loss_aux/(len(self.val_loader))
                self.best_model = self.model
            if (val_loss_aux/(len(self.val_loader))<self.best_val_loss):
                self.best_model = self.model
                self.best_val_loss = val_loss_aux/(len(self.val_loader))
                print('Best Model Yet')


        # Save the model
        save_run_history(self.best_model, self.model, self.model_save_path, self.model_name, 
                            self.x_graph, self.train_y_rec, self.train_y_kl, self.train_y_loss, hist_name='TrainHistory')
        # save_run_history(self.best_model, self.model, self.model_save_path, self.model_name, 
                            # self.x_graph, self.val_y_rec, self.val_y_kl, self.val_y_loss, hist_name='ValHistory')

        print('Network Run Complete')

    def tester(self):

        print('Model Type: %s'%self.flow_ID)
        
        # load model
        self.model.load_state_dict(torch.load(self.model_save_path + '%s.pt' %self.model_name, map_location=torch.device('cpu')))

        # load data
        self.test_loader = DataLoader(dataset=self.x_test, batch_size=self.test_batch_size, shuffle=False, drop_last=True)
        self.metTe_loader = DataLoader(dataset=self.met_test, batch_size=self.test_batch_size, shuffle=False, drop_last=True)
        self.weight_test_loader = DataLoader(dataset=self.weight_test, batch_size=self.test_batch_size, shuffle=False, drop_last=True)

        print('Starting the Testing Process ...')
        self.test_ev_rec = []
        self.test_ev_kl = []
        self.test_ev_loss = []
        for y, (x_test, met_te, wt_test) in tqdm(enumerate(zip(self.test_loader, self.metTe_loader, self.weight_test_loader))):
            if y == (len(self.test_loader)): break
            
            #Test
            te_loss, te_kl, te_eucl = test_convnet(self.model, x_test, met_te, wt_test, batch_size=self.test_batch_size)
            
            self.test_ev_loss.append(te_loss.cpu().detach().numpy())
            self.test_ev_kl.append(te_kl.cpu().detach().numpy())
            self.test_ev_rec.append(te_eucl.cpu().detach().numpy())
        # print('loss: ', test_ev_loss)
        save_npy(np.array(self.test_ev_loss), self.test_data_save_path + '%s_loss.npy' %self.model_name)
        save_npy(np.array(self.test_ev_kl), self.test_data_save_path + '%s_kl.npy' %self.model_name)
        save_npy(np.array(self.test_ev_rec), self.test_data_save_path + '%s_rec.npy' %self.model_name)
        # save_csv(data= np.array(self.test_ev_kl), filename= self.test_data_save_path + 'rec_%s.csv' %self.model_name)
        # save_csv(data= np.array(self.test_ev_rec), filename= self.test_data_save_path + 'rec1_%s.csv' %self.model_name)

        print('Testing Complete')

    # def infer(self):






        

