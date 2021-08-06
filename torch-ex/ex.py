import pickle

import torch
from torch import nn


# vocab_size = 3000
#
#
# class WordLSTM(nn.Module):
#
#     def __init__(self, n_hidden=256, n_layers=4, drop_prob=0.3, lr=0.001):
#         super().__init__()
#
#         self.drop_prob = drop_prob
#         self.n_layers = n_layers
#         self.n_hidden = n_hidden
#         self.lr = lr
#
#         self.emb_layer = nn.Embedding(vocab_size, 200)
#
#         ## define the LSTM
#         self.lstm = nn.LSTM(200, n_hidden, n_layers,
#                             dropout=drop_prob, batch_first=True)
#
#         ## define a dropout layer
#         self.dropout = nn.Dropout(drop_prob)
#
#         ## define the fully-connected layer
#         self.fc = nn.Linear(n_hidden, vocab_size)
#
#     def forward(self, x, hidden):
#         ''' Forward pass through the network.
#             These inputs are x, and the hidden/cell state `hidden`. '''
#
#         ## pass input through embedding layer
#         embedded = self.emb_layer(x)
#
#         ## Get the outputs and the new hidden state from the lstm
#         lstm_output, hidden = self.lstm(embedded, hidden)
#
#         ## pass through a dropout layer
#         out = self.dropout(lstm_output)
#
#         # out = out.contiguous().view(-1, self.n_hidden)
#         out = out.reshape(-1, self.n_hidden)
#
#         ## put "out" through the fully-connected layer
#         out = self.fc(out)
#
#         # return the final output and the hidden state
#         return out, hidden
#
#     def init_hidden(self, batch_size):
#         ''' initializes hidden state '''
#         # Create two new tensors with sizes n_layers x batch_size x n_hidden,
#         # initialized to zero, for hidden state and cell state of LSTM
#         weight = next(self.parameters()).data
#
#         # if GPU is available
#         if torch.cuda.is_available():
#             hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
#                       weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
#
#         # if GPU is not available
#         else:
#             hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
#                       weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
#
#         return hidden
#
#
# f = open("/home/rumesh/Downloads/lstm-si.pkl", "rb")
# m = pickle.load(f)
# print(type(m))


# x = torch.arange(6)
class Perceptron(nn.Module):
    """ A perceptron is one linear layer """

    def __init__(self, input_dim):
        """
        Args:
        input_dim (int): size of the input features
        """
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x_in):
        """The forward pass of the perceptron
        Args:
        x_in (torch.Tensor): an input data tensor
        x_in.shape should be (batch, num_features)
        Returns:
        the resulting tensor. tensor.shape should be (batch,).
        """
        return torch.sigmoid(self.fc1(x_in)).squeeze()


import torch.nn as nn
import torch.nn.functional as F


def describe(x):
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
        input_dim (int): the size of the input vectors
        hidden_dim (int): the output size of the first Linear layer
        output_dim (int): the output size of the second Linear layer
        """
        super(MultilayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = F.relu

    def forward(self, x_in, apply_softmax=False):
        """The forward pass of the MLP
        Args:
        x_in (torch.Tensor): an input data tensor
        x_in.shape should be (batch, input_dim)
        apply_softmax (bool): a flag for the softmax activation
        should be false if used with the cross-entropy losses
        Returns:
        the resulting tensor. tensor.shape should be (batch, output_dim)
        """
        intermediate = self.relu(self.fc1(x_in))
        output = self.fc2(intermediate)
        if apply_softmax:
            output = F.softmax(output, dim=1)
        return output


batch_size = 2  # number of samples input at once
input_dim = 3
hidden_dim = 100
output_dim = 4
# Initialize model
mlp = MultilayerPerceptron(input_dim, hidden_dim, output_dim)
print(mlp)
x_input = torch.rand(batch_size, input_dim)
describe(x_input)
y_output = mlp(x_input, apply_softmax=False)
describe(y_output)