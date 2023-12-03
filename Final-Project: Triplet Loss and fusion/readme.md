This Folder contains the code for Final project of Computational Intelligence course at FUM, released by Teaching assistant team. A sample output is provided in `./CAI` folder.
 
# Requirements
```matplotlib==3.5.2
numpy==1.21.5
Pillow==9.2.0
Pillow==10.1.0
scikit_learn==1.0.2
scipy==1.9.1
torch==1.13.1
torchvision==0.14.1
tqdm==4.64.1
```

### Data acquisition: 
download the data from https://drive.google.com/drive/folders/1TYnhCSMU1rQcAWqXYzsn-HBKYy-8IORY?usp=sharing and store it under the `./data/` directory. When loading data in `main.py`, the data proccesses automaticaly and stores the results in the `./data` directory. 
(to genereate a different dataset from the original Digit-Five dataset, download the Digit-Five dataset from [here](https://drive.google.com/open?id=1A4RJOFj4BJkmliiEL7g9WzNIDUHLxfmm) and modify `store_digit_five()`)

# Usage
Make sure the requirements are satisfied in your environment, and relevant datasets are downloaded. For using all the domains of digit-five dataset and concatenation as the fusion mode, run
```
python main.py --domains mnist mnistm usps svhn syn --fusion_mode concat
```
Some of the more important hyperparameters are listed as follows: 
- `domains`: domains to use for training and evaluation
- `margin`: margin of triplet loss
- `beta`: coefficient of triplet loss (or any other auxilary loss)
- `embed_space`: size of embedding space (fused space)
- `aux_loss`: the auxilary loss to add to cross entropy, to add new loss functions, add it to `loss_functions.py`
- `fusion_mode`: how to fuse input features and network features

# Project Description: 
## Dataset:
Projects of this course are designed around the Digit-5 dataset, a multi-domain dataset consisting of MNIST, MNIST-M, SVHN, SYN, and USPS digit datasets. Images of each domain are digits from 0 to 9 with the following dimensions: 3 x 32 x 32. Each data point in the dataset includes an image of size 3 x 32 x 32, a corresponding digit label ranging from 0 to 9, and a domain label indicating which of the 5 domains the image belongs to (MNIST, MNIST-M, SVHN, SYN, or USPS). 
(Pictures of the MNIST dataset are resized to 32x32 and the single Black and white channel is copied three times to make the picture 3x32x32.)

A couple of samples from each domain are shown in the below figure. To explore and visualize more images from the dataset, run `exploring_data/exploring_data.py`.
<img src="https://github.com/hamedrq7/CAI-Spring-2023/blob/main/exploring_data/all_domains.jpg" alt="all_domains" width="auto" height="400">

## Final-Project:
Design a model that takes a 3x32x32 image and a (256, ) feature vector for each sample, which has a digit label and a domain label and predicts the digit label using cross entropy and tripelt loss:
$$L_{T} = L_{Cross Entropy} + \lambda L_{triplet}(A, P, N)$$
where $A$ is anchor sample, $P$ and $N$ are positive and negative pairs (see `batch_hard_triplet_loss()` in `phase4.py` for more details on how positive and negative pairs are defined).
<img src="https://github.com/hamedrq7/CAI-Spring-2023/blob/main/Final-Project%3A%20Triplet%20Loss%20and%20fusion/media/triplet_info.png" width="auto" height="400">

There is a custom `DataSet` that returns image, feature vector, image label and domain label (see `exploring_data/exploring_data.py` for usage example). Follow this architecture for the model:
<img src="https://github.com/hamedrq7/CAI-Spring-2023/blob/main/Final-Project%3A%20Triplet%20Loss%20and%20fusion/media/arch.png" alt="all_domains" width="auto" height="400">

There is a "test_missing" dataset that has it's feature vector missing (set to zero). Modify the model to handle the missing input and get a good accuracy.  

- effect of Triplet loss on digits is shown in the below figure:
<img src="https://github.com/hamedrq7/CAI-Spring-2023/blob/main/Final-Project%3A%20Triplet%20Loss%20and%20fusion/media/TripletLoss%20validation%20-%20Triplet%20applied%20on%20classes%20of%20mnist%20.jpg" width="500"/>
