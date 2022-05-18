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
        # self.Data_filename = args.Data_filename
        # self.Data_bsm_filename = args.Data_bsm_filename
        # self.Met_filename = args.Met_filename
        # self.Met_bsm_filename = args.Met_bsm_filename
        self.model_name = args.model_name
        self.num_epochs = args.num_epochs
        self.num_classes = args.num_classes
        # self.training_fraction = args.training_fraction
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
        # self.channel = args.channel
        # if self.channel == 'chan1':
        #     self.num_test_ev_sm = 10000
        # elif self.channel == 'chan2a':
        #     self.num_test_ev_sm = 5868
        # elif self.channel == 'chan2b':
        #     self.num_test_ev_sm = 89000
        # elif self.channel == 'chan3':
        #     self.num_test_ev_sm = 1025333

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

        nFeat = 6

        outerdata_train = np.load("/workdir/huichi/CATHODE/preprocessed_data_6var/outerdata_train_6var.npy")
        outerdata_test = np.load("/workdir/huichi/CATHODE/preprocessed_data_6var/outerdata_test_6var.npy")

        outerdata_train = outerdata_train[outerdata_train[:,nFeat+1]==0]
        outerdata_test = outerdata_test[outerdata_test[:,nFeat+1]==0]

        data_train = outerdata_train[:,1:nFeat+1]
        print('shape of data_train: ', data_train.shape)
        data_test = outerdata_test[:,1:nFeat+1]
        print('shape of data_test: ', data_test.shape)

        data = np.concatenate((data_train, data_test), axis=0)
        print('shape of data: ', data.shape)

        cond_data_train = outerdata_train[:,0]
        print('shape of cond_train', cond_data_train.shape)
        cond_data_test = outerdata_test[:,0]
        print('shape of cond_test', cond_data_test.shape)

        cond_data = np.concatenate((cond_data_train, cond_data_test), axis=0)
        print('shape of data: ', cond_data.shape)


        # scalar_x = StandardScaler()
        # scalar_x.fit(data)
        # data = scalar_x.transform(data)
        # self.scalar_x = scalar_x

        # scalar_cond = StandardScaler()
        # cond_data = np.reshape(cond_data, [-1, 1])
        # scalar_cond.fit(cond_data)
        # cond_data = scalar_cond.transform(cond_data)
        # self.scalar_cond = scalar_cond

        max = np.empty(nFeat)
        for i in range(0,data.shape[1]):
            max[i] = np.max(np.abs(data[:,i]))
            if np.abs(max[i]) > 0: 
                data[:,i] = data[:,i]/max[i]
            else:
                pass

        self.data = data
        self.x_max = max

        cond_max = np.max(np.abs(cond_data))
        if np.abs(cond_max) > 0:
            cond_data = cond_data/cond_max
        else:
            pass

        self.cond_data = cond_data
        self.N_bkg_SB = cond_data.shape[0]


        trainsize = outerdata_train.shape[0]
        self.trainsize = trainsize
        
        x_train = data[:trainsize]
        x_test = data[trainsize:]
        y_train = cond_data[:trainsize]
        y_test = cond_data[trainsize:]

        image_size = x_train.shape[1]
        original_dim = image_size
        x_train = np.reshape(x_train, [-1, original_dim])
        x_test = np.reshape(x_test, [-1, original_dim])
        
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        
        y_train = np.reshape(y_train, [-1, 1])
        y_test = np.reshape(y_test, [-1, 1])
        
        y_train = y_train.astype('float32')
        y_test = y_test.astype('float32')

        
        self.x_train = x_train
        self.met_train = y_train 
        
        # Val data
        self.x_val = x_test
        self.met_val = y_test
        



    def trainer(self):
        self.train_loader = DataLoader(dataset = self.x_train, batch_size = self.batch_size, shuffle=True)
        self.metTr_loader = DataLoader(dataset = self.met_train, batch_size = self.batch_size, shuffle=True)
        # self.weight_train_loader = DataLoader(dataset = self.weight_train, batch_size = self.batch_size, shuffle=False, drop_last=True)

        self.val_loader = DataLoader(dataset = self.x_val, batch_size = self.batch_size, shuffle=False)
        self.metVa_loader = DataLoader(dataset = self.met_val, batch_size = self.batch_size, shuffle=False)
        # self.weight_val_loader = DataLoader(dataset = self.weight_val, batch_size = self.batch_size, shuffle=False, drop_last=True)

        # to store training history
        self.x_graph = []
        # self.train_y_rec = []
        self.train_y_kl = []
        self.train_y_loss = []
        
        # self.val_y_rec = []
        self.val_y_kl = []
        self.val_y_loss = []

        # print('Model Parameter: ', self.model)
        print('Model Type: %s'%self.flow_ID)
        print('Initiating training, validation processes ...')
        for epoch in range(self.num_epochs):
            self.x_graph.append(epoch)
            print('Starting to train ...')

            # adjust learning rate
            epoch1 = 240
            epoch2 = 120

            if epoch < epoch1*4:
                itr = epoch // epoch1
                self.learning_rate = 0.001/(2**itr)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            else:
                itr = 4 + (epoch-epoch1*4) // epoch2
                self.learning_rate = 0.001/(2**itr)
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-6, nesterov=True)


            # training
            tr_loss_aux = 0.0
            tr_kl_aux = 0.0
            tr_rec_aux = 0.0
            self.train_z_mu = np.empty(0)
            self.train_z_var = np.empty(0)
            # encoder_z_mean = []
            # encoder_z_std = []


            for y, (x_train, met_tr) in tqdm(enumerate(zip(self.train_loader, self.metTr_loader))):
                if y == (len(self.train_loader)): break

                z_mu, z_var, tr_loss, tr_kl, self.model = train_convnet(self.model, x_train, self.optimizer, batch_size=self.batch_size, beta=self.beta)
                
                tr_loss_aux += float(tr_loss)
                tr_kl_aux += float(tr_kl)
                
                # print("TTTTTTTTTT: ", z_mu.cpu().detach().numpy().size)
                # encoder_z_mean.append(z_mu)
                # encoder_z_std.append(z_var)
                if self.train_z_mu.shape[0] == 0:
                    self.train_z_mu = z_mu.cpu().detach().numpy()
                    self.train_z_var = z_var.cpu().detach().numpy()
                else:
                    self.train_z_mu = np.concatenate((self.train_z_mu, z_mu.cpu().detach().numpy()))
                    self.train_z_var = np.concatenate((self.train_z_var, z_var.cpu().detach().numpy()))
                # tr_rec_aux += tr_eucl

            print('Moving to validation stage ...')
            # validation
            val_loss_aux = 0.0
            val_kl_aux = 0.0
            val_rec_aux = 0.0

            for y, (x_val, met_va) in tqdm(enumerate(zip(self.val_loader, self.metVa_loader))):
                if y == (len(self.val_loader)): break
                
                #Test
                _, val_loss, val_kl = test_convnet(self.model, x_val, met_va, batch_size=self.batch_size, beta=self.beta)

                val_loss_aux += float(val_loss)
                val_kl_aux += float(val_kl)
                # val_rec_aux += val_eucl

            self.train_y_loss.append(tr_loss_aux/(len(self.train_loader)))
            self.train_y_kl.append(tr_kl_aux/(len(self.train_loader)))
            
            # print("TTTTTTTTTTTTB: ", encoder_z_mean)
            # self.train_z_mu.append(encoder_z_mean.cpu().detach().numpy())
            # self.train_z_var.append(encoder_z_std.cpu().detach().numpy())
            # self.train_y_rec.append(tr_rec_aux.cpu().detach().numpy()/(len(self.train_loader)))
                
            self.val_y_loss.append(val_loss_aux/(len(self.val_loader)))
            self.val_y_kl.append(val_kl_aux/(len(self.val_loader)))
            # self.val_y_rec.append(val_rec_aux/(len(self.val_loader)))
                
            print('Epoch: {} -- Train loss: {}  -- Val loss: {}'.format(epoch, 
                                                                         tr_loss_aux/(len(self.train_loader)), 
                                                                         val_loss_aux/(len(self.val_loader))))
            if (epoch == 0):
                self.best_val_loss = val_loss_aux/(len(self.val_loader))
                self.best_model = self.model
                
                self.best_train_z_mu = self.train_z_mu
                self.best_train_z_var = self.train_z_var
            if (val_loss_aux/(len(self.val_loader))<self.best_val_loss):
                self.best_model = self.model
                self.best_val_loss = val_loss_aux/(len(self.val_loader))
                
                self.best_train_z_mu = self.train_z_mu
                self.best_train_z_var = self.train_z_var
                # print("latent_mean shape: ", np.array(self.train_z_mu).shape)
                print('Best Model Yet')


        # Save the model
        # print("GGGGGGGGGG:  ", self.train_z_mu)
        # print("GGGGGGGGGG:  ", self.train_z_mu.shape)
        # print("BBBBBGGGGGGGGGG:  ", self.best_train_z_mu)


        print("Save latent info.")
        np.save('/workdir/huichi/NF-C-VAE/model_save/best_latent_mean_noCond_6var.npy', self.best_train_z_mu)
        np.save('/workdir/huichi/NF-C-VAE/model_save/best_latent_std_noCond_6var.npy', self.best_train_z_var)
        # np.save('/workdir/huichi/NF-C-VAE/model_save/latent_mean.npy', self.train_z_mu)
        # np.save('/workdir/huichi/NF-C-VAE/model_save/latent_std.npy', self.train_z_var)
        save_run_history(self.best_model, self.model, self.model_save_path, self.model_name, 
                            self.x_graph, self.train_y_kl, self.train_y_loss, hist_name='TrainHistory')
        # save_run_history(self.best_model, self.model, self.model_save_path, self.model_name, 
                            # self.x_graph, self.val_y_rec, self.val_y_kl, self.val_y_loss, hist_name='ValHistory')
        save_npy(np.array(self.train_y_loss), self.data_save_path + '%s_train_loss.npy' %self.model_name)
        save_npy(np.array(self.train_y_kl), self.data_save_path + '%s_train_kl.npy' %self.model_name)
        save_npy(np.array(self.val_y_loss), self.data_save_path + '%s_val_loss.npy' %self.model_name)
        save_npy(np.array(self.val_y_kl), self.data_save_path + '%s_val_kl.npy' %self.model_name)
        

        print('Network Run Complete')


    def event_generater_SB(self):
        self.model.load_state_dict(torch.load(self.model_save_path + 'BEST_%s.pt' %self.model_name, map_location=torch.device('cpu')))
        self.model.eval()
        with torch.no_grad():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            best_z_mu = np.load("/workdir/huichi/NF-C-VAE/model_save/best_latent_mean_noCond_6var.npy", allow_pickle=True)
            best_z_logvar = np.load("/workdir/huichi/NF-C-VAE/model_save/best_latent_std_noCond_6var.npy", allow_pickle=True)

            # reshape to (nEvent, latent_dim)
            # best_z_mu = np.concatenate(best_z_mu, axis=0)
            # best_z_logvar = np.concatenate(best_z_logvar, axis=0)

            best_z_var = np.exp(best_z_logvar)
            best_z_std = np.sqrt(best_z_var)

            z_samples = np.empty([self.N_bkg_SB, self.latent_dim])

            l=0
            for i in range(0,self.N_bkg_SB):
                for j in range(0,self.latent_dim):
                    z_samples[l,j] = np.random.normal(best_z_mu[i%self.trainsize,j], 0.05+best_z_std[i%self.trainsize,j])
                    # z_samples[l,j] = np.random.normal(0,1)
                l=l+1
                
            z_samples_tensor = torch.from_numpy(z_samples.astype('float32')).to(device)
            cond_data_tensor = torch.from_numpy(np.reshape(self.cond_data, [-1, 1]).astype('float32')).to(device)

            new_events = self.model.decode(z_samples_tensor).data.cpu().numpy()

            for i in range(0,new_events.shape[1]):
                new_events[:,i]=new_events[:,i]*self.x_max[i]
            # new_events = self.scalar_x.inverse_transform(new_events)

            np.savetxt('/workdir/huichi/NF-C-VAE/data_save/LHCO2020_B-VAE_events_6var.csv', new_events)

        print("Done generating SB events.")


    def tester(self):

        print('Model Type: %s'%self.flow_ID)
        
        # load model
        self.model.load_state_dict(torch.load(self.model_save_path + 'BEST_%s.pt' %self.model_name, map_location=torch.device('cpu')))

        # load data
        # self.test_loader = DataLoader(dataset=np.reshape(self.data, [-1, self.data.shape[1]]).astype('float32'), batch_size=self.batch_size, shuffle=False)
        # self.metTe_loader = DataLoader(dataset=np.reshape(self.cond_data, [-1, 1]).astype('float32'), batch_size=self.batch_size, shuffle=False)
        
        self.test_loader = DataLoader(dataset=self.x_train, batch_size=self.batch_size, shuffle=False)
        self.metTe_loader = DataLoader(dataset=self.met_train, batch_size=self.batch_size, shuffle=False)

        # self.weight_test_loader = DataLoader(dataset=self.weight_test, batch_size=self.test_batch_size, shuffle=False, drop_last=True)

        print('Starting the Testing Process ...')
        # self.test_ev_rec = []
        # self.test_ev_kl = []
        # self.test_ev_loss = []
        self.new_events = []

        for y, (x_test, met_te) in tqdm(enumerate(zip(self.test_loader, self.metTe_loader))):
            if y == (len(self.test_loader)): break
            
            #Test
            x_decoded, _, __ = test_convnet(self.model, x_test, met_te, batch_size=self.batch_size, beta=self.beta)
            self.new_events.append(x_decoded.cpu().detach().numpy())
            
            # self.test_ev_loss.append(te_loss.cpu().detach().numpy())
            # self.test_ev_kl.append(te_kl.cpu().detach().numpy())
            # self.test_ev_rec.append(te_eucl.cpu().detach().numpy())
        # print('loss: ', test_ev_loss)
        # save_npy(np.array(self.test_ev_loss), self.test_data_save_path + '%s_loss.npy' %self.model_name)
        # save_npy(np.array(self.test_ev_kl), self.test_data_save_path + '%s_kl.npy' %self.model_name)
        # save_npy(np.array(self.test_ev_rec), self.test_data_save_path + '%s_rec.npy' %self.model_name)
        # save_csv(data= np.array(self.test_ev_kl), filename= self.test_data_save_path + 'rec_%s.csv' %self.model_name)
        # save_csv(data= np.array(self.test_ev_rec), filename= self.test_data_save_path + 'rec1_%s.csv' %self.model_name)
        self.new_events = np.concatenate(self.new_events, axis=0)
        for i in range(0,self.new_events.shape[1]):
            self.new_events[:,i]=self.new_events[:,i]*self.x_max[i]
        # self.new_events = self.scalar_x.inverse_transform(self.new_events)

        np.savetxt('/workdir/huichi/NF-C-VAE/data_save/gen_train_noCond.csv', self.new_events)

        print('Testing Complete')

    # def infer(self):






        

