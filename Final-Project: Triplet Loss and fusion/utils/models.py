import torch
import torch.nn as nn
import torch.nn.functional as F

class fusion_model(nn.Module):
    def __init__(self, fusion_mode: str, image_embedding_space: int, do_bn_on_features: bool, dropout: bool) -> None:
        """
        fusion_mode: how to fuse given input features with embeddings produced by image encoder
        image_embedding_space: dimension of image embeddings produced by image encoder
        do_bn_on_features: if true, adds a batch norm layer on the input features 
        dropout: bool, if true, adds dropout from [512] to [84] layer. (the layer before softmax layer)
        """
        super(fusion_model, self).__init__()
        self.do_bn_on_features = do_bn_on_features
        self.fusion_mode = fusion_mode
        self.dropout = dropout
        self.FEATURES_SPACE = 256
        self.flatten = nn.Flatten()
       
        if do_bn_on_features: 
            self.features_batch_norm = nn.BatchNorm1d(self.FEATURES_SPACE)
        
        self.img_encoder = lenet_color(embed_space=image_embedding_space)

        if fusion_mode == 'concat': 
            self.fc1 = nn.Linear(image_embedding_space+self.FEATURES_SPACE, 84)
        elif fusion_mode == 'additive' or fusion_mode == 'multiplicative': 
            assert image_embedding_space == self.FEATURES_SPACE, 'for multiplicative and additive fusion, feature space and image embedding must have same dimensions.'
            self.fc1 = nn.Linear(self.FEATURES_SPACE, 84)

        if self.dropout: 
            self.dropout_layer = nn.Dropout(p=0.5)

        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(84, 10)
        
    def forward(self, image_input, features):
        """
        image_input: of size (N, 3, 32, 32)
        features: of size (N, self.FEATURE_SPACE) - set to 256

        returns 
              image embeddings produced by image encoder       (embed),
              activations of last layer (before softmax layer) (deep_feats),
              softmax scores of the classification layer       (nn.functional.log_softmax(x, dim=1))
        """
        embed = self.img_encoder(image_input)

        if self.do_bn_on_features: 
            features = self.features_batch_norm(features)

        # with torch.no_grad():
        #     print('embeddings: (mean, max, min, std)', torch.mean(embed).item(), torch.max(embed).item(), torch.min(embed).item(), torch.std(embed).item())
        #     print('features:   (mean, max, min, std)', torch.mean(features).item(), torch.max(features).item(), torch.min(features).item(), torch.std(features).item())
        
        if self.fusion_mode == 'concat': 
            x = torch.cat([embed, features], dim=1)
        
        elif self.fusion_mode == 'additive': 
            x = torch.add(embed, features)

        elif self.fusion_mode == 'multiplicative':
            x = torch.mul(embed, features)

        ### tensor fusion
        # elif self.fusion_mode == 'tensor': 
        # A_expanded = embed.unsqueeze(2)  # Shape: (N, D, 1)
        # B_expanded = features.unsqueeze(1)  # Shape: (N, 1, D)
        # # Perform element-wise multiplication
        # x = torch.matmul(A_expanded, B_expanded)
        # x = self.flatten(x)
        # # large x, [batch_Size, 65536]

        # dropout before passing 512 features to 84 features
        if self.dropout: 
            x = self.dropout_layer(x)
        
        deep_feats = self.fc1(x)
        x = self.prelu1(deep_feats)
        
        x = self.fc2(x)

        return embed, deep_feats, nn.functional.log_softmax(x, dim=1)

class lenet_color(nn.Module):
    def __init__(self, embed_space: int):
        super(lenet_color, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.AvgPool2d(2)
        
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.AvgPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, 3)
        self.prelu3 = nn.PReLU()

        self.fc1 = nn.Linear(64*3*3, embed_space)
        self.prelu3 = nn.PReLU()
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.prelu1(y)
        y = self.pool1(y)

        y = self.conv2(y)
        y = self.prelu2(y)
        y = self.pool2(y)
        
        y = self.conv3(y)
        y = self.prelu3(y)
        y = y.view(y.shape[0], -1)

        y = self.fc1(y)
        y = self.prelu3(y)
        
        return y 


class largeCNN(nn.Module):
    def __init__(self, embed_space=256, num_classes=10):
        super(largeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 1024)
        self.bn2_fc = nn.BatchNorm1d(1024)

        self.fc3 = nn.Linear(1024, embed_space)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), 8192)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = self.bn2_fc(self.fc2(x))
        
        x = self.fc3(F.relu(x))
        
        return x