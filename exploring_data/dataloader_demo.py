from out import get_data_loaders
import torch 

def loaders_demo():
    full_dataloaders, _ = get_data_loaders(
    {'train': './data/1000_train_mnistmnistmsvhnsynusps.npz', 
     'test': './data/1000_test_mnistmnistmsvhnsynusps.npz',
     },
    batch_size= 64) 
    print(full_dataloaders.keys())
    
    for phase in ['train', 'test', 'test_missing']: 
        print(f'{phase} data...')
        for batch_indx, (images, features, domain_labels, digit_labels) in enumerate(full_dataloaders[phase]):
            print(f'{batch_indx}-th batch')
            print('images shape: ', images.shape)
            print('features shape: ', features.shape)
            if phase == 'test_missing': 
                print('in test-missing dataloaders, since the features are not available, features are filled with zeros', torch.sum(features))
            print('domain labels freq: ', torch.unique(domain_labels, return_counts=True))
            print('digit labels freq: ', torch.unique(digit_labels, return_counts=True))
            print()
            break

loaders_demo()