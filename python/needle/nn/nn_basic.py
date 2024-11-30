"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


# class Linear(Module):
#     def __init__(
#         self, in_features, out_features, bias=True, device=None, dtype="float32"
#     ):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features

#         ### BEGIN YOUR SOLUTION
#         raise NotImplementedError()
#         ### END YOUR SOLUTION

#     def forward(self, X: Tensor) -> Tensor:
#         ### BEGIN YOUR SOLUTION
#         raise NotImplementedError()
#         ### END YOUR SOLUTION


# class Flatten(Module):
#     def forward(self, X):
#         ### BEGIN YOUR SOLUTION
#         raise NotImplementedError()
#         ### END YOUR SOLUTION


# class ReLU(Module):
#     def forward(self, x: Tensor) -> Tensor:
#         ### BEGIN YOUR SOLUTION
#         raise NotImplementedError()
#         ### END YOUR SOLUTION

# class Sequential(Module):
#     def __init__(self, *modules):
#         super().__init__()
#         self.modules = modules

#     def forward(self, x: Tensor) -> Tensor:
#         ### BEGIN YOUR SOLUTION
#         raise NotImplementedError()
#         ### END YOUR SOLUTION


# class SoftmaxLoss(Module):
#     def forward(self, logits: Tensor, y: Tensor):
#         ### BEGIN YOUR SOLUTION
#         raise NotImplementedError()
#         ### END YOUR SOLUTION


# class BatchNorm1d(Module):
#     def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
#         super().__init__()
#         self.dim = dim
#         self.eps = eps
#         self.momentum = momentum
#         ### BEGIN YOUR SOLUTION
#         raise NotImplementedError()
#         ### END YOUR SOLUTION

#     def forward(self, x: Tensor) -> Tensor:
#         ### BEGIN YOUR SOLUTION
#         raise NotImplementedError()
#         ### END YOUR SOLUTION
class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(fan_in = in_features, fan_out = out_features, dtype = dtype, device = device).reshape((in_features, out_features)))
        self.bias = None
        if bias:
          self.bias = Parameter(init.kaiming_uniform(fan_in = out_features, fan_out = 1, dtype = dtype, device = device).reshape((1, out_features)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = X @ self.weight
        
        if self.bias is not None:
            y = ops.add(y, self.bias.reshape((1, self.out_features)).broadcast_to(y.shape))
        return y
        ### END YOUR SOLUTION

class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        val = 1
        for dim in X.shape[1:]:
            val *= dim
        return X.reshape((X.shape[0], val))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x) 
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        y_one_hot = init.one_hot(logits.shape[1], y)
        logsumexp = ops.logsumexp(logits, axes=(1,))
        loss = logsumexp - ops.summation(logits * y_one_hot, axes=(1,))
        return ops.summation(loss) / logits.shape[0]
        ### END YOUR SOLUTION

class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = Tensor(init.zeros(dim, device=device, dtype=dtype), requires_grad=False)
        self.running_var = Tensor(init.ones(dim, device=device, dtype=dtype), requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N, D = x.shape
        if self.training:
            b_mean = ops.reshape((ops.summation(x, axes=0) / N), (1, D))  
            m_b = ops.broadcast_to(b_mean, x.shape) 
            batch_var = ops.reshape(ops.summation((x - m_b) * (x - m_b), axes=0) / N , (1, D))  
            std = (ops.broadcast_to(batch_var, x.shape) + self.eps) ** 0.5
            non_b_mean = ops.reshape(b_mean, (D,))
            non_batch_var = ops.reshape(batch_var, (D,))
            self.running_mean = ((1 - self.momentum) * self.running_mean) + (self.momentum * non_b_mean.data)
            self.running_var = ((1 - self.momentum) * self.running_var) + (self.momentum * non_batch_var.data)
        else:
            running_mean = ops.reshape(self.running_mean, (1, D))
            running_var = ops.reshape(self.running_var, (1, D))
            m_b = ops.broadcast_to(running_mean, x.shape) 
            var_broadcast = ops.broadcast_to(running_var, x.shape)
            std = (var_broadcast + self.eps) ** 0.5
            x_normalized = (x - m_b) / std
      
        out = ((x - m_b) / std) * ops.broadcast_to(ops.reshape(self.weight, (1, D)), x.shape)  + ops.broadcast_to(ops.reshape(self.bias, (1, D)), x.shape)

        return out
        ### END YOUR SOLUTION  
class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))
class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, dtype=dtype, device=device))
        self.bias = Parameter(init.zeros(dim, dtype=dtype, device=device))
        ### END YOUR SOLUTION
        
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean = ops.broadcast_to(ops.reshape(ops.summation(x, axes=1) / x.shape[1], (x.shape[0], 1)), x.shape)
        var = ops.reshape(ops.summation((x - mean) * (x - mean), axes=1) / x.shape[1], (x.shape[0], 1)) 
        val = (x-mean) / ops.broadcast_to((var + self.eps) ** 0.5, x.shape)
        weight = ops.broadcast_to(ops.reshape(self.weight, (1, x.shape[1])), x.shape)
        bias = ops.broadcast_to(ops.reshape(self.bias, (1, x.shape[1])), x.shape)
        return val * weight + bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=1-self.p, dtype="float32")
            mask = mask / (1 - self.p)
            return x * mask
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION

