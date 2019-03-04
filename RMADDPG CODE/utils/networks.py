import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out


class RNNNetwork(nn.Module):
    """
    RNN network. Experimenting with using it as both policy and value function.
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):

        super(RNNNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim + hidden_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        # The RNN has two fc hidden layers
        # Input to hidden unit 1
        self.i2h1 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        # input to hidden unit 2
        # self.i2h2 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.i2h2 = nn.Linear(hidden_dim, hidden_dim)
        self.h2o = nn.Linear(input_dim + hidden_dim, out_dim)
        self.nonlin = nonlin

        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.h2o.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        # Preprocess X into obs for t, t-1, t-2 . . . t-n
        # modifications for batch
        hist_tminus_0 = torch.tensor(np.zeros((X.shape[0], 18)), dtype=torch.float)
        hist_tminus_1 = torch.tensor(np.zeros((X.shape[0], 18)), dtype=torch.float)
        hist_tminus_2 = torch.tensor(np.zeros((X.shape[0], 18)), dtype=torch.float)
        hist_tminus_3 = torch.tensor(np.zeros((X.shape[0], 18)), dtype=torch.float)
        hist_tminus_4 = torch.tensor(np.zeros((X.shape[0], 18)), dtype=torch.float)
        hist_tminus_5 = torch.tensor(np.zeros((X.shape[0], 18)), dtype=torch.float)

        for n in range(X.shape[0]):
            hist_tminus_0[n] = X[n][0:18]
            hist_tminus_1[n] = X[n][18:36]
            hist_tminus_2[n] = X[n][36:54]
            hist_tminus_3[n] = X[n][54:72]
            hist_tminus_4[n] = X[n][72:90]
            hist_tminus_5[n] = X[n][90:108]

        init_memory = torch.tensor(np.zeros((X.shape[0],self.hidden_dim)), dtype=torch.float)

        X = torch.cat((hist_tminus_5, init_memory), 1)      # oldest history and intialization
        X = self.i2h2(self.i2h1(X))                         # Now, X is of shape hidden_dim
        X = torch.cat((hist_tminus_4, X), 1)
        X = self.i2h2(self.i2h1(X))
        X = torch.cat((hist_tminus_3, X), 1)
        X = self.i2h2(self.i2h1(X))
        X = torch.cat((hist_tminus_2, X), 1)
        X = self.i2h2(self.i2h1(X))
        X = torch.cat((hist_tminus_1, X), 1)
        X = self.i2h2(self.i2h1(X))
        X = torch.cat((hist_tminus_0, X), 1)
        X = self.h2o(X)
        return X

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

