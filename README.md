# CAI-Spring-2023

## Dataset:
Projects of this course are designed around the Digit-5 dataset, a multi-domain dataset consisting of MNIST, MNIST-M, SVHN, SYN, and USPS digit datasets. Images of each domain are digits from 0 to 9 with the following dimensions: 3 x 32 x 32. Each data point in the dataset includes an image of size 3 x 32 x 32, a corresponding digit label ranging from 0 to 9, and a domain label indicating which of the 5 domains the image belongs to (MNIST, MNIST-M, SVHN, SYN, or USPS). 
(Pictures of the MNIST dataset are resized to 32x32 and the single Black and white channel is copied three times to make the picture 3x32x32.)

A couple of samples from each domain are shown in the below figure. To explore and visualize more images from the dataset, run `exploring_data/exploring_data.py`.
<img src="https://github.com/hamedrq7/CAI-Spring-2023/blob/main/exploring_data/all_domains.jpg" alt="all_domains" width="auto" height="400">



- Final-Project:
Design a model that takes a 3x32x32 image and a (256, ) feature vector for each sample, which has a digit label and a domain label and predicts the digit label using cross entropy and tripelt loss:
$$L_{T} = L_{Cross Entropy} + \lambda L_{triplet}(A, P, N)$$
where $A$ is anchor sample, $P$ and $N$ are positive and negative pairs (see `batch_hard_triplet_loss()` in `phase4.py` for more details on how positive and negative pairs are defined).
<img src="https://github.com/hamedrq7/CAI-Spring-2023/blob/main/Final-Project%3A%20Triplet%20Loss%20and%20fusion/media/triplet_info.png" width="auto" height="400">

There is a custom `DataSet` that returns image, feature vector, image label and domain label (see `exploring_data/exploring_data.py` for usage example). Follow this architecture for the model:
<img src="https://github.com/hamedrq7/CAI-Spring-2023/blob/main/Final-Project%3A%20Triplet%20Loss%20and%20fusion/media/arch.png" alt="all_domains" width="auto" height="400">

There is a "test_missing" dataset that has it's feature vector missing (set to zero). Modify the model to handle the missing input and get a good accuracy.  

