import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

floatX = 'float32'

class Network(nn.Module):
    def __init__(self, state_shape, hidden_dim, actions_dim, nc, remove_features=False):
        super(Network, self).__init__()
        self.remove_features = remove_features
        if remove_features:
            state_shape = state_shape[: -1] + [state_shape[-1] - nc + 1]
        self.state_shape = state_shape
        self.in_dim = tuple(state_shape)
        self.hidden_dim = hidden_dim
        self.actions_dim = actions_dim
        self.nc = nc
        fc_in_dim = np.prod(state_shape[0:])

        self.main = nn.Sequential(
            nn.Linear(fc_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, actions_dim),
        )

    def forward(self, states):
        bz = states.data.size(0)
        states = states.view(bz, -1)
        output = self.main(states)
        return output
