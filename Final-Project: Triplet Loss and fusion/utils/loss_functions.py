import torch
import os
import torch
from torch.optim import optimizer
from tqdm import trange
from typing import List, Dict
import numpy as np
import torch.nn as nn 



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
