# relative import hacks (sorry)
import inspect
import os 
import sys 
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) # for bash user
os.chdir(parentdir) # for pycharm user

from typing import List
from scipy.io import loadmat
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from utils.datasets_ import Dataset as tempDS
import torch.utils.data as data
from tqdm import trange
import pickle 
from numpy import savez_compressed, save, load
import torch 
from os.path import exists
from sklearn.decomposition import PCA

def get_train_test_size(input_size, available_train_size, dataset_name):
    train_size = input_size
    test_size  = int(train_size * (9/25))
    print(train_size, test_size)
    return train_size, test_size


def make_dir(path_to_save):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = np.zeros((len(labels_dense),))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        if t == 10:
            t = 0
            labels_one_hot[i] = t
        else:
            labels_one_hot[i] = t
    return labels_one_hot

base_dir = './data'
man_seed = 11

def load_svhn(size=None):
    # svhn_train = loadmat(base_dir + '/train_32x32.mat')
    # svhn_test = loadmat(base_dir + '/test_32x32.mat')
    
    svhn_train = loadmat(base_dir + '/svhn_train_32x32.mat')
    svhn_test = loadmat(base_dir + '/svhn_test_32x32.mat')
    print(svhn_train['X'].shape, svhn_train['y'].shape, svhn_test['X'].shape, svhn_test['y'].shape)

    svhn_train_im = svhn_train['X']
    svhn_train_im = svhn_train_im.transpose(3, 2, 0, 1).astype(np.float32)

    print('svhn train y shape before dense_to_one_hot->', svhn_train['y'].shape)
    svhn_label = dense_to_one_hot(svhn_train['y'])
    print('svhn train y shape after dense_to_one_hot->',svhn_label.shape)

    ### loading Test data 
    svhn_test_im = svhn_test['X']
    svhn_test_im = svhn_test_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label_test = dense_to_one_hot(svhn_test['y'])
    
    if size is None:
        svhn_train_im = svhn_train_im[:25000]
        svhn_label = svhn_label[:25000]
        svhn_test_im = svhn_test_im[:9000]
        svhn_label_test = svhn_label_test[:9000]
    else:
        train_size, test_size = get_train_test_size(size, 25000, 'svhn')
        svhn_train_im = svhn_train_im[:train_size]
        svhn_label = svhn_label[:train_size]
        svhn_test_im = svhn_test_im[:test_size]
        svhn_label_test = svhn_label_test[:test_size]

    print('svhn train X shape->',  svhn_train_im.shape)
    print('svhn train y shape->',  svhn_label.shape)
    print('svhn test X shape->',  svhn_test_im.shape)
    print('svhn test y shape->', svhn_label_test.shape)

    return svhn_train_im, svhn_label, svhn_test_im, svhn_label_test

def load_mnist(size=None, scale=True, usps=False, all_use=False):
    mnist_data = loadmat(base_dir + '/mnist_data.mat')
    if scale:
        mnist_train = np.reshape(mnist_data['train_32'], (55000, 32, 32, 1))
        mnist_test = np.reshape(mnist_data['test_32'], (10000, 32, 32, 1))
        mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
        mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)
        mnist_train = mnist_train.transpose(0, 3, 1, 2).astype(np.float32)
        mnist_test = mnist_test.transpose(0, 3, 1, 2).astype(np.float32)
        mnist_labels_train = mnist_data['label_train']
        mnist_labels_test = mnist_data['label_test']
    else:
        mnist_train = mnist_data['train_28']
        mnist_test =  mnist_data['test_28']
        mnist_labels_train = mnist_data['label_train']
        mnist_labels_test = mnist_data['label_test']
        mnist_train = mnist_train.astype(np.float32)
        mnist_test = mnist_test.astype(np.float32)
        mnist_train = mnist_train.transpose((0, 3, 1, 2))
        mnist_test = mnist_test.transpose((0, 3, 1, 2))

    train_label = np.argmax(mnist_labels_train, axis=1)
    inds = np.random.permutation(mnist_train.shape[0])
    mnist_train = mnist_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnist_labels_test, axis=1)
    
    if size is None:
        
        mnist_train = mnist_train[:25000]
        train_label = train_label[:25000]
        mnist_test = mnist_test[:9000]
        test_label = test_label[:9000]
    else:
        train_size, test_size = get_train_test_size(size, 25000, 'mnist')
        
        mnist_train = mnist_train[:train_size]
        train_label = train_label[:train_size]
        mnist_test = mnist_test[:test_size]
        test_label = test_label[:test_size]

    print('mnist train X shape->',  mnist_train.shape)
    print('mnist train y shape->',  train_label.shape)
    print('mnist test X shape->',  mnist_test.shape)
    print('mnist test y shape->', test_label.shape)

    return mnist_train, train_label, mnist_test, test_label

def load_mnistm(size=None, scale=True, usps=False, all_use=False):
    mnistm_data = loadmat(base_dir + '/mnistm_with_label.mat')
    mnistm_train = mnistm_data['train']
    mnistm_test =  mnistm_data['test']
    mnistm_train = mnistm_train.transpose(0, 3, 1, 2).astype(np.float32)
    mnistm_test = mnistm_test.transpose(0, 3, 1, 2).astype(np.float32)
    mnistm_labels_train = mnistm_data['label_train']
    mnistm_labels_test = mnistm_data['label_test']
    
    train_label = np.argmax(mnistm_labels_train, axis=1)
    inds = np.random.permutation(mnistm_train.shape[0])
    mnistm_train = mnistm_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnistm_labels_test, axis=1)
    
    if size is None:
        mnistm_train = mnistm_train[:25000]
        train_label = train_label[:25000]
        mnistm_test = mnistm_test[:9000]
        test_label = test_label[:9000]
    else:
        train_size, test_size = get_train_test_size(size, 25000, 'mnitm')
        mnistm_train = mnistm_train[:train_size]
        train_label = train_label[:train_size]
        mnistm_test = mnistm_test[:test_size]
        test_label = test_label[:test_size]
   
    print('mnist_m train X shape->',  mnistm_train.shape)
    print('mnist_m train y shape->',  train_label.shape)
    print('mnist_m test X shape->',  mnistm_test.shape)
    print('mnist_m test y shape->', test_label.shape)
    return mnistm_train, train_label, mnistm_test, test_label


def load_usps(size=None, all_use=False):
    #f = gzip.open('data/usps_28x28.pkl', 'rb')
    #data_set = pickle.load(f)
    #f.close()
    dataset  = loadmat(base_dir + '/usps_28x28.mat')
    data_set = dataset['dataset']
    img_train = data_set[0][0]
    label_train = data_set[0][1]
    img_test = data_set[1][0]
    label_test = data_set[1][1]
    inds = np.random.permutation(img_train.shape[0])
    img_train = img_train[inds]
    label_train = label_train[inds]
    
    img_train = img_train * 255
    img_test = img_test * 255
    img_train = img_train.reshape((img_train.shape[0], 1, 28, 28))
    img_test = img_test.reshape((img_test.shape[0], 1, 28, 28))


    #img_test = dense_to_one_hot(img_test)
    label_train = dense_to_one_hot(label_train)
    label_test = dense_to_one_hot(label_test)

    if size is None:
        img_train = np.concatenate([img_train, img_train, img_train, img_train], 0)[0:25000]
        label_train = np.concatenate([label_train, label_train, label_train, label_train], 0)[0:25000]
        img_test = np.concatenate([img_test, img_test, img_test, img_test], 0)[0:9000]
        label_test = np.concatenate([label_test, label_test, label_test, label_test], 0)[0:9000]

    else:
        train_size, test_size = get_train_test_size(size, 25000, 'usps')

        img_train = np.concatenate([img_train, img_train, img_train, img_train], 0)[:train_size]
        label_train = np.concatenate([label_train, label_train, label_train, label_train], 0)[:train_size]
        img_test = np.concatenate([img_test, img_test, img_test, img_test], 0)[:test_size]
        label_test = np.concatenate([label_test, label_test, label_test, label_test], 0)[:test_size]

        
    print('usps train X shape->',  img_train.shape)
    print('usps train y shape->',  label_train.shape)
    print('usps test X shape->',  img_test.shape)
    print('usps test y shape->', label_test.shape)


    return img_train, label_train, img_test, label_test


def load_syn(size=None, scale=True, usps=False, all_use=False):
    syn_data = loadmat(base_dir + '/syn_number.mat')
    syn_train = syn_data['train_data']
    syn_test =  syn_data['test_data']
    syn_train = syn_train.transpose(0, 3, 1, 2).astype(np.float32)
    syn_test = syn_test.transpose(0, 3, 1, 2).astype(np.float32)
    syn_labels_train = syn_data['train_label']
    syn_labels_test = syn_data['test_label']

    # print(syn_train.shape)

    train_label = syn_labels_train
    inds = np.random.permutation(syn_train.shape[0])
    syn_train = syn_train[inds]
    train_label = train_label[inds]
    test_label = syn_labels_test
    

    if size is None:   
        syn_train = syn_train[:25000]
        train_label = train_label[:25000]
        syn_test = syn_test[:9000]
        test_label = test_label[:9000]
    else:
        train_size, test_size = get_train_test_size(size, 25000, 'syn')
        syn_train = syn_train[:train_size]
        train_label = train_label[:train_size]
        syn_test = syn_test[:test_size]
        test_label = test_label[:test_size]

    train_label = dense_to_one_hot(train_label)
    test_label = dense_to_one_hot(test_label)

    print('syn number train X shape->',  syn_train.shape)
    print('syn number train y shape->',  train_label.shape)
    print('syn number test X shape->',  syn_test.shape)
    print('syn number test y shape->', test_label.shape)
    return syn_train, train_label, syn_test, test_label


def return_dataset(data, size=None):
    if data == 'svhn':
        train_image, train_label, \
        test_image, test_label = load_svhn(size)
    if data == 'mnist':
        train_image, train_label, \
        test_image, test_label = load_mnist(size)
    if data == 'mnistm':
        train_image, train_label, \
        test_image, test_label = load_mnistm(size)
    if data == 'usps':
        train_image, train_label, \
        test_image, test_label = load_usps(size)
    if data == 'syn':
        train_image, train_label, \
        test_image, test_label = load_syn(size)
    
    return train_image, train_label, test_image, test_label

def store_digit_five(domains: List[str], size=None, features_dim = 256):

    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dss = domains
    
    all_img = {'train': [], 'test': []}
    all_lbl = {'train': [], 'test': []}
    all_domain_lbl = {'train': [], 'test': []}

    for ds_idx, ds in enumerate(dss):
        tr_img, tr_lbl, tst_img, tst_lbl = return_dataset(ds, size)
        
        for phase in ['train', 'test']:
            if phase == 'train':
                imgs = tr_img
                lbls = tr_lbl
            else:
                imgs = tst_img
                lbls = tst_lbl

            temp_dataSet = tempDS(imgs, lbls, transform)

            imgs_preprocessed = []
            for idx in trange(temp_dataSet.__len__()):
                curr_img, curr_lbl = temp_dataSet.__getitem__(idx)
                imgs_preprocessed.append(curr_img.numpy())
            
            imgs_preprocessed = np.array(imgs_preprocessed)

            all_img[phase].append(imgs_preprocessed)
            all_lbl[phase].append(lbls)
            all_domain_lbl[phase].append(np.ones_like(lbls) * ds_idx)

    all_features = {'train': None, 'test': None}
    pca_on_train_set = None
    mu = 0.0
    std = 0.0
    max_min = 0.0

    # concat and save all 
    for phase in ['train', 'test']:
        print('phase', phase)
        all_img[phase] = np.concatenate(all_img[phase])
        all_lbl[phase] = np.concatenate(all_lbl[phase])
        all_domain_lbl[phase] = np.concatenate(all_domain_lbl[phase])

        if phase == 'train': 
            # build PCs
            pca_on_train_set = PCA(n_components=features_dim).fit(np.reshape(all_img['train'], \
                                                                             (all_img['train'].shape[0], all_img['train'].shape[1]*all_img['train'].shape[2]*all_img['train'].shape[3])))
            all_features['train'] = pca_on_train_set.transform(np.reshape(all_img['train'], \
                                                                             (all_img['train'].shape[0], all_img['train'].shape[1]*all_img['train'].shape[2]*all_img['train'].shape[3])))
            mu = np.mean(all_features['train'])
            std = np.std(all_features['train']) + 1e-12
            max_min = np.max(all_features['train']) - np.min(all_features['train']) + 1e-12
        
        else:
            all_features['test'] = pca_on_train_set.transform(np.reshape(all_img['test'], \
                                                                             (all_img['test'].shape[0], all_img['test'].shape[1]*all_img['test'].shape[2]*all_img['test'].shape[3])))
        
        all_features[phase] = (all_features[phase] - mu) / std

        print(all_img[phase].shape)
        print(all_lbl[phase].shape)
        print(all_domain_lbl[phase].shape)
        print(all_features[phase].shape)

        domains_str = ''.join(dss)

        name_to_save = ''
        if size is None:
            name_to_save = f'{phase}_{domains_str}.npz'
        else:
            name_to_save = f'{size}_{phase}_{domains_str}.npz'

        if phase == 'train':
            savez_compressed(f'./data/{name_to_save}', train_imgs = all_img[phase], train_features = all_features[phase], train_digit_labels=all_lbl[phase], train_domain_labels=all_domain_lbl[phase])
        else:
            savez_compressed(f'./data/{name_to_save}', test_imgs = all_img[phase], test_features = all_features[phase], test_digit_labels=all_lbl[phase], test_domain_labels=all_domain_lbl[phase])
  
def load_ds(filename: str, phase: str, features_missing: bool = False):
    """
    phase either 'train' or 'test' or 'test_missing'
    """
    data = np.load(f'{filename}')
    imgs = data[f'{phase}_imgs']
    digit_labels = data[f'{phase}_digit_labels']
    domain_labels = data[f'{phase}_domain_labels']
    if features_missing: 
        features = np.zeros_like(data[f'{phase}_features'])
    else:    
        features = data[f'{phase}_features']

    print(f'{phase} digit labels count: ')
    unique, counts = np.unique(digit_labels, return_counts=True)
    print(counts)
    print(f'{phase} domain labels count: ')
    unique, counts = np.unique(domain_labels, return_counts=True)
    print(counts)
    
    # no shuffling 
    # inds = np.random.permutation(imgs.shape[0])
    # imgs = imgs[inds]
    # digit_labels = digit_labels[inds]
    # domain_labels = domain_labels[inds]

    print(imgs.shape)
    print('images: (mean, max, min)', np.mean(imgs), np.max(imgs), np.min(imgs))

    print(digit_labels.shape)
    print(domain_labels.shape)
    print(features.shape)
    print('features: (mean, max, min)', np.mean(features), np.max(features), np.min(features))

    return imgs, digit_labels, domain_labels, features, np.unique(domain_labels).shape[0]


class customDataset(data.Dataset):
    def __init__(self, data, domain_labels, digit_labels, features) -> None:
        super().__init__()
        self.data = data
        self.domain_labels = domain_labels
        self.digit_labels = digit_labels
        self.features = features
    
    def __getitem__(self, index):
        """
        return data, domain label, digit label
        ---> change it to: 
        return data, features, domain label, digit label
        """
        # return torch.FloatTensor(self.data[index]), torch.tensor(self.domain_labels[index]).type(torch.int64), torch.tensor(self.digit_labels[index]).type(torch.int64)
        # ---> changed it
        return torch.FloatTensor(self.data[index]), \
                torch.FloatTensor(self.features[index]), \
                    torch.tensor(self.domain_labels[index]).type(torch.int64), torch.tensor(self.digit_labels[index]).type(torch.int64)
        

        # torch.FloatTensor(np.ones(256) * self.digit_labels[index] / 10)

    def __len__(self): 
        return self.data.shape[0]


def get_data_loaders(domains: List[str], batch_size: int = 128, size: int = None):
    
    file_names = {'train': '', 'test': ''}
    if size is None:
        domain_str = "".join(domains)
        file_names['train'] = f'./data/train_{domain_str}.npz'
        file_names['test'] = f'./data/test_{domain_str}.npz'
        
    else:
        domain_str = "".join(domains)
        file_names['train'] = f'./data/{size}_train_{domain_str}.npz'
        file_names['test'] = f'./data/{size}_test_{domain_str}.npz'

    print('datafiles to read: ', file_names)

    files_exist = exists(file_names['train']) and exists(file_names['test'])

    if not files_exist:
        store_digit_five(domains, size=size)
    
    full_datasets = {'train': None, 'test': None, 'test_missing': None}
    full_dataloaders = {'train': None, 'test': None, 'test_missing': None, 'train_size': None, 'test_size': None, 'test_missing_size': None}

    for phase in ['train', 'test', 'test_missing']:
        if phase == 'test_missing': 
            imgs, digit_labels, domain_labels, features, num_domains = load_ds(file_names['test'], 'test', features_missing = True)
        else:
            imgs, digit_labels, domain_labels, features, num_domains = load_ds(file_names[phase], phase)

        full_datasets[phase] = customDataset(imgs, domain_labels, digit_labels, features)
        full_dataloaders[phase] = data.DataLoader(full_datasets[phase], batch_size=batch_size, shuffle=True) # (phase=='train')
        full_dataloaders[f'{phase}_size'] = full_datasets[phase].__len__()

    return full_dataloaders, num_domains

