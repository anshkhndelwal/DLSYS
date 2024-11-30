import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

class ConvBN(ndl.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, device=None, dtype="float32"):   
        self.model = nn.Sequential(nn.Conv(in_channels, out_channels, kernel_size, stride=stride, device=device, dtype=dtype),
        nn.BatchNorm2d(out_channels, device=device, dtype=dtype),
        nn.ReLU())
        
    def forward(self, x):
        return self.model.forward(x)

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.model = nn.Sequential(
            ConvBN(3, 16, 7, 4, device, dtype),
            ConvBN(16, 32, 3, 2, device, dtype),
            nn.Residual(nn.Sequential(ConvBN(32, 32, 3, 1, device, dtype),ConvBN(32, 32, 3, 1, device, dtype),)),
            ConvBN(32, 64, 3, 2, device, dtype),
            ConvBN(64, 128, 3, 2, device, dtype),
            nn.Residual(nn.Sequential(ConvBN(128, 128, 3, 1, device, dtype),ConvBN(128, 128, 3, 1, device, dtype),)),
            nn.Flatten(),
            nn.Linear(128, 128, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 10, device=device, dtype=dtype),
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.model.forward(x)
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding = nn.Embedding(output_size, embedding_size, device = device, dtype = dtype)
        if seq_model == 'lstm':
            self.sequential = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        elif seq_model == 'rnn':
            self.sequential = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        x = self.embedding(x)
        x, h = self.sequential(x, h)
        result = self.linear(x.reshape((x.shape[0] * x.shape[1], x.shape[2])))
        return result, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)