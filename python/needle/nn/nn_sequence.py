"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module, ReLU


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.power_scalar(1+ops.exp(-x), -1)
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.W_ih = Parameter(init.rand( input_size, hidden_size, low=-(1 / hidden_size**0.5), high=(1 / hidden_size**0.5), device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-(1 / hidden_size**0.5), high=(1 / hidden_size**0.5),device=device, dtype=dtype, requires_grad=True))

        if bias:
            self.bias_ih =  Parameter(init.rand(hidden_size, low=-(1 / hidden_size**0.5), high=(1 / hidden_size**0.5),device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-(1 / hidden_size**0.5), high=(1 / hidden_size**0.5),device=device, dtype=dtype, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_ih = None
        self.hidden_size = hidden_size
        self.activation = nonlinearity
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = init.zeros(X.shape[0], self.hidden_size,dtype=X.dtype, device=X.device)
        Z = X @ self.W_ih + h @ self.W_hh
        if self.bias_ih:
            Z += self.bias_ih.reshape((1, self.hidden_size)).broadcast_to((X.shape[0], self.hidden_size)) + self.bias_hh.reshape((1, self.hidden_size)).broadcast_to((X.shape[0], self.hidden_size))
        if self.activation=='tanh':
            return ops.tanh(Z)
        return ops.relu(Z)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.rnn_cells = []
        self.rnn_cells.append(RNNCell(input_size, hidden_size, bias=bias,nonlinearity=nonlinearity, device=device, dtype=dtype))
        for i in range(num_layers - 1):
          self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias=bias,nonlinearity=nonlinearity, device=device, dtype=dtype))

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        val = ops.split(X, axis=0) 
        h_n = []
        if h0:
            h0 = ops.split(h0, axis=0)
        for layer in range(self.num_layers):
            if h0:
                val1 = ops.tuple_get_item(h0, layer)
            else:
                val1 = None
            H_l = [] 
            for i in range(X.shape[0]):
                val1 = self.rnn_cells[layer](ops.tuple_get_item(val, i), val1) 
                H_l.append(val1)
            val = ops.make_tuple(*H_l)
            h_n.append(val1)          
        return ops.stack(val, axis=0), ops.stack(h_n, axis=0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sigmoid = Sigmoid()
        self.tanh = ops.tanh

        self.W_ih = Parameter(init.rand(input_size,hidden_size * 4,low=-(1 / hidden_size ** 0.5),high=(1 / hidden_size ** 0.5),device=device, dtype=dtype,requires_grad=True,))
        self.W_hh = Parameter(init.rand(hidden_size,hidden_size * 4,low=-(1 / hidden_size ** 0.5),high=(1 / hidden_size ** 0.5),device=device, dtype=dtype,requires_grad=True,))
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size * 4,low=-(1 / hidden_size ** 0.5),high=(1 / hidden_size ** 0.5),device=device, dtype=dtype,requires_grad=True,))
            self.bias_hh = Parameter(init.rand(hidden_size * 4,low=-(1 / hidden_size ** 0.5),high=(1 / hidden_size ** 0.5),device=device, dtype=dtype,requires_grad=True,))
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        
        bs, _ = X.shape

        if h:
            h0, c0 = h
        else:
            h0 = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
            c0 = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)

        x = X @ self.W_ih + h0 @ self.W_hh 

        if self.bias_ih:
            x += self.bias_ih.broadcast_to((bs, self.hidden_size * 4))
            x += self.bias_hh.broadcast_to((bs, self.hidden_size * 4))

        x_split = tuple(ops.split(x, axis=1))

        i = self.sigmoid(ops.stack(x_split[0 : self.hidden_size], axis=1))
        f = self.sigmoid(ops.stack(x_split[self.hidden_size : 2 * self.hidden_size], axis=1))
        g = self.tanh(ops.stack(x_split[2 * self.hidden_size : 3 * self.hidden_size], axis=1))
        o = self.sigmoid(ops.stack(x_split[3 * self.hidden_size : 4 * self.hidden_size], axis=1))
        c = f * c0 + i * g
        h = o * self.tanh(c)
        return o * self.tanh(c), f * c0 + i * g
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = []
        for k in range(num_layers):
            if k == 0:
              input_dim = input_size
            else:
              input_dim = hidden_size
            self.lstm_cells.append(LSTMCell(input_size=input_dim, hidden_size=hidden_size, bias=bias, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h:
            h0, c0 = h
        else:
            h0 = init.zeros(self.num_layers, X.shape[1], self.hidden_size, device=X.device, dtype=X.dtype)
            c0 = init.zeros(self.num_layers, X.shape[1], self.hidden_size, device=X.device, dtype=X.dtype)

        X_t, H_n, C_n = ops.split(X, axis=0), ops.split(h0, axis=0), ops.split(c0, axis=0)
        H_t = []

        for t in range(X.shape[0]):
            x_t = X_t[t] 
            h_n, c_n = [], []

            for l in range(self.num_layers):
                lstm_cell = self.lstm_cells[l]
                h_tl, c_tl = lstm_cell(x_t, (H_n[l], C_n[l])) 
                x_t = h_tl

                h_n.append(h_tl)
                c_n.append(c_tl)

                if l == self.num_layers - 1:
                    H_t.append(h_tl)
            
            H_n, C_n = h_n, c_n
        
        return ops.stack(H_t, axis=0), (ops.stack(H_n, axis=0), ops.stack(C_n, axis=0))
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(num_embeddings,embedding_dim,mean=0,std=1,device=device, dtype=dtype,requires_grad=True,))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        one_hot = init.one_hot(self.num_embeddings, x, device=x.device, dtype=x.dtype)
        one_hot = one_hot.reshape((x.shape[0] * x.shape[1], self.num_embeddings))
        return (one_hot @ self.weight).reshape((x.shape[0], x.shape[1], self.embedding_dim))
        ### END YOUR SOLUTION