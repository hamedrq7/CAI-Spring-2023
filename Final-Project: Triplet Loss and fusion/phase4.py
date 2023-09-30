import torch
import os
import torch
from torch.optim import optimizer
from tqdm import trange
from typing import List, Dict
import numpy as np
from d5 import get_data_loaders
import torch.nn as nn 
from tqdm import trange
from utils import *
from models import *

class myTripletLoss(nn.Module):
    def __init__(self, device, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = torch.tensor(margin)
        self.device = device

    def all_diffs(self, a, b):
        # a, b -> [N, d]
        # return -> [N, N, d]
        return a[:, None] - b[None, :]

    def euclidean_dist(self, embed1, embed2):
        # embed1, embed2 -> [N, d]
        # return [N, N] -> # get a square matrix of all diffs, diagonal is zero
        diffs = self.all_diffs(embed1, embed2) 
        t1 = torch.square(diffs)
        t2 = torch.sum(t1, dim=-1)
        return torch.sqrt(t2 + 1e-12)

    def batch_hard_triplet_loss(self, dists, labels):
        # labels -> [N, 1]
        # dists -> [N, N], square mat of all distances, 
        # dists[i, j] is distance between sample[i] and sample[j]

    
        same_identity_mask = torch.eq(labels[:, None], labels[None, :]) 
        # [N, N], same_mask[i, j] = True when sample i and j have the same label

        negative_mask = torch.logical_not(same_identity_mask)
        # [N, N], negative_mask[i, j] = True when sample i and j have different label

        positive_mask = torch.logical_xor(same_identity_mask, torch.eye(labels.shape[0], dtype=torch.bool).to(self.device))
        # [N, N], same as same_identity mask, except diagonal is zero

        furthest_positive, _ = torch.max(dists * (positive_mask.int()), dim=1)

        closest_negative = torch.zeros_like(furthest_positive)
        for i in range(dists.shape[0]):
            closest_negative[i] = torch.min(dists[i, :][negative_mask[i, :]])    

        diff = furthest_positive - closest_negative

        return torch.max(diff + self.margin, torch.tensor(0.0))
    
    def forward(self, embeddings, labels):
        dists = self.euclidean_dist(embeddings, embeddings)
        losses = self.batch_hard_triplet_loss(dists, labels)

        return torch.mean(losses)

class Exp:
    def __init__(self, use_gpu: bool = True) -> None:
        self.lr: float
        self.full_dataloaders: Dict[str, DataLoader]
        self.opt: nn.Module
        self.model: nn.Module
        self.use_gpu = use_gpu
        
        self.triplet_loss: myTripletLoss
        # seeds
        rand_seed = 11
        torch.manual_seed(rand_seed)
        torch.cuda.manual_seed(rand_seed)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.use_gpu else "cpu")

        self.img_path = 'imgs'
        self.exp_name = 'default'
        self.path_save_model = 'models'

    def set_env(self, margin: float, 
                 embed_space: int, beta: float,   
                 model_name: str,
                 fusion_mode: str,
                 bn_on_features: bool,
                 dropout: bool,
                 lr: float,
                 exp_dir: str,
                 full_dataloaders: Dict[str, DataLoader], optimzier_str: str = 'sgd'):
        """
        margin: passed on to tripletloss, margin for triplet loss
        embed_space: passed on to model, embed_space for outputs of image encoder which is the input of triplet loss
        beta: coeff that multiplies the triplet loss and sums up with CE
        model_name: str,
        fusion_mode: passed on to model, how to fuse given input features with embeddings produced by image encoder
        """
        self.lr = lr
        self.full_dataloaders = full_dataloaders
        self.margin = margin
        self.embed_space = embed_space
        self.beta = beta

        ### set model
        if model_name == 'fusion_model':
            self.model = fusion_model(fusion_mode, image_embedding_space=embed_space, do_bn_on_features=bn_on_features, dropout=dropout)
            self.model_name = model_name
            self.model = self.model.to(self.device)

        else:
            print('invalid model name.')

        print(self.model)

        ### set optimizer
        if optimzier_str == 'sgd':
            self.opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        elif optimzier_str == 'adam':
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            
        ### set CE loss for digits
        self.ce_loss = nn.NLLLoss()

        ### set triplet loss
        # 0.2 for mnist
        self.triplet_loss = myTripletLoss(self.device, margin)
        
        self.exp_name = f'opti_{optimzier_str}-fusion_mode_{fusion_mode}-beta_{beta}-margin_{margin}-bn_on_features_{bn_on_features}-dropout_{dropout}-embed_space_{embed_space}-model_name_{model_name}-lr_{lr}'        
        self.img_path = f'{exp_dir}/imgs/{self.exp_name}'
        self.path_save_model = f'{exp_dir}/models/{self.exp_name}'

    def train(self, num_epochs: int, beta2: float = 1.0):
        print('training...')
        print('exp ', self.exp_name)

        self.acc_history = {'train': [], 'test': [], 'test_missing': []}
        self.loss_history = {'train': [], 'test': [], 'test_missing': []}
        self.triplet_loss_history = {'train': [], 'test': [], 'test_missing': []}
        self.ce_loss_history = {'train': [], 'test': [], 'test_missing': []}

        for epoch in trange(num_epochs):

            for phase in ['train', 'test', 'test_missing']: 
                running_loss = 0.0
                running_loss_CE = 0.0
                running_loss_T = 0.0
                running_corrects = 0
                epoch_features = []
                epoch_labels = []
                epoch_domain_labels = []

                if phase == 'train': self.model.train()
                else: self.model.eval()

                for batch_indx, (inputs, features, domain_labels, digit_labels) in enumerate(self.full_dataloaders[phase]):
                    # _, freqs = torch.unique(domain_labels, return_counts=True)
                    # print(freqs)
                    
                    self.opt.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):                    
                        inputs = inputs.to(self.device)
                        features = features.to(self.device)
                        domain_labels = domain_labels.to(self.device)
                        digit_labels = digit_labels.to(self.device)

                        embeddings, deep_feats, outputs = self.model(inputs, features)
                        _, preds = torch.max(outputs, dim=1)
                    
                        # triplet loss on domains
                        t_loss = self.triplet_loss(embeddings, domain_labels)
                        c_loss = self.ce_loss(outputs, digit_labels)
                    
                        loss = self.beta * t_loss + beta2 * c_loss
                    
                        if phase == 'train':
                            loss.backward()
                            self.opt.step()

                    running_loss += loss.item()
                    running_loss_CE += c_loss.item()
                    running_loss_T += t_loss.item()
                    running_corrects += torch.sum(preds == digit_labels)
                    
                    epoch_features.append(deep_feats.cpu().detach().numpy())
                    epoch_labels.append(digit_labels.cpu().detach().numpy())
                    epoch_domain_labels.append(domain_labels.cpu().detach().numpy())

                epoch_loss    = (running_loss / self.full_dataloaders[f'{phase}_size'])
                epoch_loss_CE = (running_loss_CE / self.full_dataloaders[f'{phase}_size'])
                epoch_loss_T  = (running_loss_T / self.full_dataloaders[f'{phase}_size'])
                epoch_loss /= (batch_indx+1)
                epoch_loss_CE /= (batch_indx+1)
                epoch_loss_T /= (batch_indx+1)

                self.loss_history[phase].append(epoch_loss)
                self.ce_loss_history[phase].append(epoch_loss_CE)
                self.triplet_loss_history[phase].append(epoch_loss_T)

                epoch_acc = (running_corrects.double() / self.full_dataloaders[f'{phase}_size']).cpu().numpy()
                self.acc_history[phase].append(epoch_acc)

                print(f'{phase} total loss: {epoch_loss}, CE loss: {epoch_loss_CE}, triplet_loss: {epoch_loss_T}, acc: {epoch_acc}')

                epoch_features = np.concatenate(epoch_features)
                epoch_labels = np.concatenate(epoch_labels)
                epoch_domain_labels = np.concatenate(epoch_domain_labels)

                plot_feats = False
                if plot_feats:
                    # plot labels based on the true labels 
                    if beta2 != 0.0:
                        custom_plot_features(features=epoch_features, labels=epoch_labels,  
                                            path = f'{self.img_path}/feats', 
                                            title=f'phase_{phase}-epoch_{epoch}-deep_{self.exp_name}')
                    else: 
                        custom_plot_features(features=epoch_features, labels=epoch_domain_labels,  
                                            path = f'{self.img_path}/feats', 
                                            title=f'phase_{phase}-epoch_{epoch}-deep_feats{self.exp_name}')

        plot_loss = True
        plot_acc = True
        if plot_loss and plot_acc:
            custom_plot_training_stats(self.acc_history, self.loss_history, ['train', 'test', 'test_missing'], title=f'{self.exp_name}', dir=self.img_path)
            custpm_plot_loss(self.ce_loss_history, ['train', 'test', 'test_missing'], loss_name='cross entropy loss', title=f'{self.exp_name}', dir=self.img_path)
            custpm_plot_loss(self.triplet_loss_history, ['train', 'test', 'test_missing'], loss_name='triplet loss', title=f'{self.exp_name}', dir=self.img_path)
            
        save_model = True
        if save_model: 
            mkdir(self.path_save_model)
            torch.save(self.model, f'{self.path_save_model}/{self.exp_name}.pt')

    def load_model(self, model_name_to_load: str):
        if model_name_to_load is None:  
            self.model = torch.load(f'{self.path_save_model}/{self.exp_name}.pt')
        else:
            self.model = torch.load(f'{self.path_save_model}/{model_name_to_load}.pt')

        self.model = self.model.to(self.device)

# sorted(['mnistm', 'mnist', 'usps', 'svhn', 'syn'])
# full_dataloaders, num_domains = get_data_loaders(sorted(['mnistm', 'mnist', 'usps', 'svhn', 'syn']), batch_size= 64, size=1000) 

# exp = Exp(True)
# exp.set_env(margin=0.5, beta=0.0, embed_space=256, model_name='fusion_model', fusion_mode='concat', bn_on_features=False, dropout=False, lr=0.01, full_dataloaders=full_dataloaders, optimzier_str='sgd')
# exp.train(10)

# size 2
# margin 2
# beta 2
# fusion_mode 2
# bn 2

# Exp Adam
for size in [12000]: 
    for fusion_mode in ['concat']: 
        for bn in [False]: 
            for margin in [0.001]: # good for digits on 5 datasets: 0.01, 0.005, 0.001, 
                for beta in [0.0]: # 0.0, 0.05
                    for drp in [False]:
                        for opt in ['sgd', 'adam']: 
                            full_dataloaders, num_domains = get_data_loaders(sorted(['mnistm', 'mnist', 'usps', 'svhn', 'syn']), batch_size= 64, size=size) 
                            exp = Exp(True)
                            exp.set_env(margin=margin, beta=beta, embed_space=256, 
                                model_name='fusion_model', fusion_mode=fusion_mode, 
                                bn_on_features=bn, dropout=drp, lr=0.001, 
                                full_dataloaders=full_dataloaders, optimzier_str=opt,
                                exp_dir='Adam vs SGD')
                            exp.train(10)

for size in [12000]: 
    for fusion_mode in ['concat']: 
        for bn in [True]: 
            for margin in [0.001]: # good for digits on 5 datasets: 0.01, 0.005, 0.001, 
                for beta in [0.0]: # 0.0, 0.05
                    for drp in [True]:
                        for opt in ['sgd']: # , 'adam' 
                            full_dataloaders, num_domains = get_data_loaders(sorted(['mnistm', 'mnist', 'usps', 'svhn', 'syn']), batch_size= 64, size=size) 
                            exp = Exp(True)
                            exp.set_env(margin=margin, beta=beta, embed_space=256, 
                                model_name='fusion_model', fusion_mode=fusion_mode, 
                                bn_on_features=bn, dropout=drp, lr=0.001, 
                                full_dataloaders=full_dataloaders, optimzier_str=opt,
                                exp_dir='Adam vs SGD')
                            exp.train(10)
# Exp1.2
# for size in [12000]: 
#     for fusion_mode in ['concat']: 
#         for bn in [True]: 
#             for margin in [0.001]: # good for digits on 5 datasets: 0.01, 0.005, 0.001, 
#                 for beta in [0.0]: # 0.0, 0.05
#                     for drp in [False]: 
#                         full_dataloaders, num_domains = get_data_loaders(sorted(['mnistm', 'mnist', 'usps', 'svhn', 'syn']), batch_size= 64, size=size) 
#                         exp = Exp(True)
#                         exp.set_env(margin=margin, beta=beta, embed_space=256, 
#                             model_name='fusion_model', fusion_mode=fusion_mode, 
#                             bn_on_features=bn, dropout=drp, lr=0.01, 
#                             full_dataloaders=full_dataloaders, optimzier_str='sgd',
#                             exp_dir='exp1.2: effect of BN on baseline models, no Dropout')
#                         exp.train(10)
    
# Exp1.3
# for size in [12000]: 
#     for fusion_mode in ['concat']: 
#         for bn in [False]: 
#             for margin in [0.001]: # good for digits on 5 datasets: 0.01, 0.005, 0.001, 
#                 for beta in [0.0, 0.05]: 
#                     for drp in [True]: 
#                         full_dataloaders, num_domains = get_data_loaders(sorted(['mnistm', 'mnist', 'usps', 'svhn', 'syn']), batch_size= 64, size=size) 
#                         exp = Exp(True)
#                         exp.set_env(margin=margin, beta=beta, embed_space=256, 
#                             model_name='fusion_model', fusion_mode=fusion_mode, 
#                             bn_on_features=bn, dropout=drp, lr=0.01, 
#                             full_dataloaders=full_dataloaders, optimzier_str='sgd',
#                             exp_dir='exp1.3: effect of dropout on baseline models, no BN')
#                         exp.train(10)
    

# Exp1.4
# for size in [12000]: 
#     for fusion_mode in ['concat']: 
#         for bn in [True]: 
#             for margin in [0.001]: # good for digits on 5 datasets: 0.01, 0.005, 0.001, 
#                 for beta in [0.0, 0.05]: 
#                     for drp in [True]: 
#                         full_dataloaders, num_domains = get_data_loaders(sorted(['mnistm', 'mnist', 'usps', 'svhn', 'syn']), batch_size= 64, size=size) 
#                         exp = Exp(True)
#                         exp.set_env(margin=margin, beta=beta, embed_space=256, 
#                             model_name='fusion_model', fusion_mode=fusion_mode, 
#                             bn_on_features=bn, dropout=drp, lr=0.01, 
#                             full_dataloaders=full_dataloaders, optimzier_str='sgd',
#                             exp_dir='exp1.4: effect of both dropout and BN on baseline models')
#                         exp.train(10)
    
