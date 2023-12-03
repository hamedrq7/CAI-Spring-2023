import torch
import os
import torch
from torch.optim import optimizer
from tqdm import trange
from typing import List, Dict
import numpy as np
import torch.nn as nn 
from tqdm import trange
import argparse
import sys
import json

# relative import hacks (sorry)
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) # for bash user
os.chdir(parentdir) # for pycharm user

from utils.d5 import get_data_loaders
from utils.log import Logger, Timer, save_model, save_vars
from utils.utils import *
from utils.models import *
import utils.loss_functions as loss_functions

parser = argparse.ArgumentParser(description='cai')

parser.add_argument('--experiment', type=str, default='CAI', metavar='E',
                    help='directory to save the results')
parser.add_argument('-l','--domains', nargs='+', default=['mnistm', 'mnist', 'usps', 'svhn', 'syn'], 
                    help='Domains to use for train and evaluation (must choose at least one from mnist, mnistm, usps, svhn, syn)', 
                    # required=True
                    )
parser.add_argument('--seed', type=int, default=70, metavar='S',
                    help='random seed (default: 70)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA use')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='batch size for data (default: 128)')
parser.add_argument('--data-size', type=int, default=12000, metavar='N',
                    help='amount of data to use from each domains (default: 12000)')
parser.add_argument('--margin', type=float, default=0.001,
                    help='margin for triplet loss: (0.001)')
parser.add_argument('--beta', type=float, default=0.0,
                    help='coefficient for triplet loss: (0.0)')
parser.add_argument('--embed_space', type=int, default=256,
                    help='size of the embedding space: (256)')
parser.add_argument('--model_name', type=str, default='fusion_model',
                    help='model name to use, a class with the given name must exist in models.py')
parser.add_argument('--aux_loss', type=str, default='myTripletLoss',
                    help='auxilary loss to use for domains')

# Add tensor fusion with the trick used in "https://aclanthology.org/D17-1115.pdf"
parser.add_argument('--fusion_mode', type=str, default='concat',
                    help='how to fuse representations with given features',
                    choices=['concat', 'additive', 'multiplicative'])
parser.add_argument('--bn_feats', action='store_true', default=False,
                    help='if True, adds a batch norm layer in input space of given features, default: (False) ')
parser.add_argument('--dropout_feats', action='store_true', default=False,
                    help='if True, adds a dropout layer in the fusion layer, default: (False) ')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate, default: (0.001)')
parser.add_argument('--optim', type=str, default='adam', metavar='M',
                    choices=['adam', 'sgd'],
                    help='optimizer (default: adam) - the sgd option is with momentum of 0.9')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')


args = parser.parse_args()

print(args)

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

        ### set auxilary loss
        # 0.2 for mnist
        lossC = getattr(loss_functions, args.aux_loss)
        self.triplet_loss = lossC(self.device, margin)


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

# Reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Save outputs
runPath = args.experiment
mkdir(runPath)
sys.stdout = Logger('{}/run.log'.format(runPath))
print('Expt:', runPath)
command_line_args = sys.argv
command = ' '.join(command_line_args)
print(f"The command that ran this script: {command}")

# save args to run
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
torch.save(args, '{}/args.rar'.format(runPath))


if __name__ == '__main__':
    with Timer('CAI-Final-project') as t:      
        full_dataloaders, num_domains = get_data_loaders(sorted(args.domains), batch_size= args.batch_size, size=args.data_size) 
        exp = Exp(True if torch.cuda.is_available() and not args.no_cuda else False)

        exp.set_env(margin=args.margin, beta=args.beta, embed_space=args.embed_space, 
            model_name=args.model_name, fusion_mode=args.fusion_mode, 
            bn_on_features=args.bn_feats, dropout=args.dropout_feats, lr=args.lr, 
            full_dataloaders=full_dataloaders, optimzier_str=args.optim,
            exp_dir=args.experiment)
        exp.train(args.epochs)

