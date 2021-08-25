# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable



class sANN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=10, num_layers=2):
        super().__init__()

        self.out = nn.Sequential(
            nn.Linear(input_size,hidden_size[0]),
            nn.dropout(0.5)
            nn.Linear(hidden_size[0],hidden_size[1]),
            nn.dropout(0.5)
            nn.Linear(hidden_size[1], num_classes)
            nn.dropout(0.5)
            nn.Sigmoid())

    def forward(self, features, init_hidden=None):
        """
        Args:
            features: [seq_len, 1, 100] (compressed pool5 features)
        Return:
            scores [seq_len, 1]
        """
        scores = self.out(features.squeeze(1))

        return scores


class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.nz = input_size
        self.ngf = 64
        self.nc = 3
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Summarizer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.s_ann = sANN(input_size, hidden_size, num_layers=num_layers)
        
    def forward(self, image_features, uniform=False):
        # Apply weights
        if not uniform:
            # [seq_len, 1]
            scores = self.s_ANN(image_features)

            # [seq_len, 1, hidden_size]
            weighted_features = image_features * scores.view(-1, 1, 1)
        else:
            scores = None
            weighted_features = image_features

        return scores, weighted_features


if __name__ == '__main__':

    pass
