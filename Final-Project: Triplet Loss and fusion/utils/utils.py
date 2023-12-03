import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os 
import torchvision
import random
from torch.utils.data import DataLoader


def mkdir(dir: str):
  if not os.path.exists(dir):
    os.makedirs(dir)

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def custom_plot_features(features, labels, title: str, path: str='images/feats', save_only=True):  
    if features.shape[1] > 2: 
        # tsne 
        # features = TSNE(n_components=2).fit_transform(features)
        features = PCA(n_components=2).fit_transform(features)
    plt.clf()
    # plt.set_aspect('equal', adjustable='box')
    fig, ax = plt.subplots()
    fig.set_dpi(150)
    fig.set_size_inches(12, 10)

    # ax.set_aspect('equal', 'box')
    
    # ax.set_figure(figsize=(12, 10), dpi=150)

    c = ['black', '#0000ff', '#990000', '#00ffff', '#ffff00',
                        '#ff00ff', '#009900', '#999900', '#00ff00', '#009999']    
  
    if features.shape[0] < 2000: marker_size = 4
    elif features.shape[0] < 11000: marker_size = 3
    else: marker_size = 1

    for class_num in range(10):
        ax.plot(features[labels==class_num, 0], 
                    features[labels==class_num, 1],
                    '.', ms=marker_size, c=c[class_num], alpha=0.4, label=class_num)

        
    # ax.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right', markerscale=3)
    ax.legend()
    ax.set_title(title)
    
    # ax.ylim(-200, 200)
    # ax.xlim(-200, 200)
    ax.axis('square')
 
    mkdir(path)
    plt.savefig(f'{path}/{title}.jpg')
    
    if not save_only:
        plt.show()
    
    plt.clf()

def get_mnist_data(data_loader_seed: int, batch_size = 128, selected_classes=list(np.arange(10)), tr_10k: bool=False):
    trainset = torchvision.datasets.MNIST('MNIST', download=True, train=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    
    testset = torchvision.datasets.MNIST('MNIST', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

    train_idxs = torch.where(torch.isin(trainset.targets, torch.asarray(selected_classes)))[0]
    test_idxs = torch.where(torch.isin(testset.targets, torch.asarray(selected_classes)))[0]

    def seed_worker(worker_id):
        # worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(data_loader_seed)
        random.seed(data_loader_seed)

    g = torch.Generator()
    g.manual_seed(data_loader_seed)
      
    tr_size = train_idxs.shape[0]
    te_size = test_idxs.shape[0]
    
    if tr_10k:
        indices = torch.arange(1024*10)
        tr_size = 1024*10
        trainset = torch.utils.data.Subset(trainset, indices)

    
    # have to set shuffle to False, since SubsetRandomSampler already shuffles data
    full_data_loaders = {'train': DataLoader(trainset, batch_size=batch_size, 
                                            shuffle=False, # 
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            # sampler=SubsetRandomSampler(train_idxs, g),
                                            ),
                        'train_size': tr_size,
                        'test': DataLoader(testset, batch_size=batch_size, shuffle=False,
                                           # sampler=SubsetRandomSampler(test_idxs),
                                            ),
                        'test_size': te_size,
                                           }
    
    input_size = 28 * 28
    num_classes = 10

    return full_data_loaders, input_size, num_classes, batch_size


def custpm_plot_loss(loss_hist, phase_list, loss_name: str, title: str, dir: str): 
    fig, ax1 = plt.subplots(nrows = 1, ncols = 1, figsize=[7, 6], dpi=100)

    for phase in phase_list: 
        lowest_loss_x = np.argmin(np.array(loss_hist[phase]))
        lowest_loss_y = loss_hist[phase][lowest_loss_x]
        
        ax1.annotate("{:.4f}".format(lowest_loss_y), [lowest_loss_x, lowest_loss_y])
        ax1.plot(loss_hist[phase], '-x', label=f'{phase} loss', markevery = [lowest_loss_x])

        ax1.set_xlabel(xlabel='epochs')
        ax1.set_ylabel(ylabel='loss')

        ax1.grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
        ax1.legend()
        ax1.label_outer()

    fig.suptitle(f'{title}')

    mkdir(dir)
    plt.savefig(f'{dir}/loss_{loss_name}.jpg')
    plt.clf()


def custom_plot_training_stats(acc_hist, loss_hist, phase_list, title: str, dir: str): 
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize=[14, 6], dpi=100)


    for phase in phase_list: 
        lowest_loss_x = np.argmin(np.array(loss_hist[phase]))
        lowest_loss_y = loss_hist[phase][lowest_loss_x]
        
        ax1.annotate("{:.4f}".format(lowest_loss_y), [lowest_loss_x, lowest_loss_y])
        ax1.plot(loss_hist[phase], '-x', label=f'{phase} loss', markevery = [lowest_loss_x])

        ax1.set_xlabel(xlabel='epochs')
        ax1.set_ylabel(ylabel='loss')

        ax1.grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
        ax1.legend()
        ax1.label_outer()

    # acc: 
    for phase in phase_list:
        highest_acc_x = np.argmax(np.array(acc_hist[phase]))
        highest_acc_y = acc_hist[phase][highest_acc_x]
        
        ax2.annotate("{:.4f}".format(highest_acc_y), [highest_acc_x, highest_acc_y])
        ax2.plot(acc_hist[phase], '-x', label=f'{phase} acc', markevery = [highest_acc_x])

        ax2.set_xlabel(xlabel='epochs')
        ax2.set_ylabel(ylabel='acc')

        ax2.grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
        ax2.legend()
        #ax2.label_outer()

    fig.suptitle(f'{title}')

    mkdir(dir)
    plt.savefig(f'{dir}/acc_loss.jpg')
    plt.clf()