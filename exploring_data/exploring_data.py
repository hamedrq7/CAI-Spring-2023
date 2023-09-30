import torch
import numpy as np
import matplotlib.pyplot as plt
from out import get_data_loaders

NUM_SAMPLES_FROM_DIGITS = 5
NUM_DOMAINS = 5
NUM_DIGITS = 10
SEED = 141+1

DOMAIN_NAME_DICT = {
    0: 'mnist',
    1: 'mnistm',
    2: 'svhn',
    3: 'syn',
    4: 'usps' 
} 

# get data loaders
# 1000_test_mnistmnistmsvhnsynusps.npz
# 1000_train_mnistmnistmsvhnsynusps.npz
full_dataloaders, _ = get_data_loaders(
    {'train': './data/1000_train_mnistmnistmsvhnsynusps.npz', 
     'test': './data/1000_test_mnistmnistmsvhnsynusps.npz',
     },
    batch_size= 64, init_seed=SEED) 

# utils
def get_samples(dataloader, num_samples: int) -> np.ndarray: 
    samples = [
        [ [], [], [], [], [], [], [], [], [], [] ],
        [ [], [], [], [], [], [], [], [], [], [] ],
        [ [], [], [], [], [], [], [], [], [], [] ],
        [ [], [], [], [], [], [], [], [], [], [] ],
        [ [], [], [], [], [], [], [], [], [], [] ],
    ]
    
    for batch_indx, (images, _, domain_labels, digit_labels) in enumerate(dataloader): 
        # images are normalized using mean=(0.5, 0.5, 0.5) and std=(0.5, 0.5, 0.5),
        # so images habve been normalized using: image = image - mean / std
        # to plot images we have to undo the normalization 
        images = images * 0.5 + 0.5
        
        for img_indx, curr_image in enumerate(images):
            
            if len(samples[domain_labels[img_indx]][digit_labels[img_indx]]) < num_samples:
                samples[domain_labels[img_indx]][digit_labels[img_indx]].append(curr_image.numpy())

    # convert samples to numpy array
    return np.array(samples)



# plot NUM_SAMPLES_FROM_DIGITS of each domain together: 
samples = get_samples(full_dataloaders['train'], NUM_SAMPLES_FROM_DIGITS)

fig_height = 10
fig_width = fig_height * (NUM_DOMAINS*NUM_SAMPLES_FROM_DIGITS) / NUM_DIGITS
fig, axs = plt.subplots(NUM_DIGITS, NUM_DOMAINS*NUM_SAMPLES_FROM_DIGITS, figsize=(fig_width, fig_height), 
                        # gridspec_kw=dict(hspace=0.0)
                        gridspec_kw={'height_ratios':[1]*10}
                        )

for i in range(NUM_DIGITS): 
    for dom in range(NUM_DOMAINS):
        for j in range(NUM_SAMPLES_FROM_DIGITS):  
            image = np.transpose(samples[dom][i][j], (1, 2, 0))  # Transpose the image to (32, 32, 3)
            axs[i, j+dom*NUM_SAMPLES_FROM_DIGITS].imshow(image)
            axs[i, j+dom*NUM_SAMPLES_FROM_DIGITS].axis('off')

plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
plt.tight_layout()
plt.savefig('exploring_data/all_domains.jpg')
plt.clf()


# plot 10 samples of each domain separately
samples = get_samples(full_dataloaders['train'], 10)

for domain in DOMAIN_NAME_DICT.keys(): 
    print(f'plotting samples from domain {DOMAIN_NAME_DICT[domain]}')

    fig, axs = plt.subplots(10, 10, figsize=(10, 10), gridspec_kw=dict(hspace=0.0))

    for i in range(10):  
        for j in range(10):  
            image = np.transpose(samples[domain][i][j], (1, 2, 0))  # Transpose the image to (32, 32, 3)
            axs[j, i].imshow(image)
            axs[j, i].axis('off')

    plt.tight_layout()
    plt.title(f'{DOMAIN_NAME_DICT[domain]}')
    plt.savefig(f'exploring_data/{DOMAIN_NAME_DICT[domain]}.jpg')
    plt.clf()
