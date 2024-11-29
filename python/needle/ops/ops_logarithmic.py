from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

# class LogSoftmax(TensorOp):
#     def compute(self, Z):
#         ### BEGIN YOUR SOLUTION
#         raise NotImplementedError()
#         ### END YOUR SOLUTION

#     def gradient(self, out_grad, node):
#         ### BEGIN YOUR SOLUTION
#         raise NotImplementedError()
#         ### END YOUR SOLUTION


# def logsoftmax(a):
#     return LogSoftmax()(a)


# class LogSumExp(TensorOp):
#     def __init__(self, axes: Optional[tuple] = None):
#         self.axes = axes

#     def compute(self, Z):
#         ### BEGIN YOUR SOLUTION
#         raise NotImplementedError()
#         ### END YOUR SOLUTION

#     def gradient(self, out_grad, node):
#         ### BEGIN YOUR SOLUTION
#         raise NotImplementedError()
#         ### END YOUR SOLUTION


# def logsumexp(a, axes=None):
#     return LogSumExp(axes=axes)(a)





# from typing import Optional
# from ..autograd import NDArray
# from ..autograd import Op, Tensor, Value, TensorOp
# from ..autograd import TensorTuple, TensorTupleOp

# from .ops_mathematic import *

# import numpy as array_api


class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, axis=1, keepdims=True)
        log_sum_exp = array_api.log(array_api.sum(array_api.exp(Z - max_Z), axis=1, keepdims=True))
        return Z - max_Z - log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        Z = node.inputs[0].realize_cached_data()
        softmax = Tensor(array_api.exp(self.compute(Z)))
        grad_Z = out_grad - softmax * out_grad.sum(axes=(-1,)).reshape((-1, 1))
        return grad_Z

def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION

        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        log_Z = array_api.log(array_api.sum(array_api.exp(Z - max_Z.broadcast_to(Z.shape)), axis=self.axes, keepdims=True))
        assert max_Z.shape == log_Z.shape
        return (log_Z + max_Z).squeeze()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        max_Z = array_api.max(Z.realize_cached_data(), axis=self.axes, keepdims=True)
        exp_Z = exp(Z - max_Z.broadcast_to(Z.shape))
        Z_sum_exp = broadcast_to(exp_Z.sum(axes=self.axes).reshape(max_Z.shape), exp_Z.shape)
        out_grad = broadcast_to(out_grad.reshape(max_Z.shape), exp_Z.shape)
        return out_grad * exp_Z / Z_sum_exp
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
