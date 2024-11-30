"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *
from ..init import ones, zeros

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
      if a.device != b.device:
          b = b.to(a.device)
      return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        if a.device != b.device:
          b = b.to(a.device)
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** b
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_grad = out_grad * rhs * power(lhs, rhs - 1)
        rhs_grad = out_grad * log(lhs) * power(lhs, rhs)
        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        lhs_grad = out_grad * power_scalar(lhs, self.scalar - 1) * self.scalar
        return lhs_grad
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / (rhs**2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        order = list(range(len(a.shape)))
        if self.axes:
            order[self.axes[0]], order[self.axes[1]] = order[self.axes[1]], order[self.axes[0]]
        else:
            order = order[:-2] +[order[-1],order[-2]]
        return a.permute(tuple(order))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)

class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs_shape = node.inputs[0].shape
        return reshape(out_grad, shape=lhs_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)

# class BroadcastTo(TensorOp):
#     def __init__(self, shape):
#         self.shape = shape

#     def compute(self, a):
#       ### BEGIN YOUR SOLUTION
#         if a.shape == self.shape:
#             return a
#         return array_api.broadcast_to(a, self.shape).compact()
#         ### END YOUR SOLUTION

#     def gradient(self, out_grad, node):
#       ### BEGIN YOUR SOLUTION
#         val = node.inputs[0]
#         len_a = list(val.shape)
#         axes = []
#         len_a += [1] * (len(self.shape) - len(len_a))
#         for i, sh in enumerate(self.shape):
#             if sh != len_a[i] or i >= len(len_a):
#                 axes.append(i)
#         axes = tuple(axes)
#         return (reshape(summation(out_grad, axes), len_a))
#       ### END YOUR SOLUTION
class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        val = node.inputs[0]
        len_a = len(val.shape)

        axes = []
        for i in range(len_a, len(self.shape)):
          axes.append(i - len_a)

        for i in reversed(range(len_a)):
          if val.shape[i] == 1:
            axes.append(i - len_a)
        
        a_grad = summation(out_grad, axes=tuple(axes))
        return reshape(a_grad, val.shape)



def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if isinstance(self.axes, (list, tuple)) and len(self.axes) > 1:
          for axis in (sorted(self.axes, reverse = True)):
              a = a.sum(axis = axis)
          return a
        return a.sum(axis=self.axes)

        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs_shape = node.inputs[0].shape
        
        if self.axes is None:
            return broadcast_to(out_grad, lhs_shape)
        
        new_shape = list(lhs_shape)
        if isinstance(self.axes, int):
            new_shape[self.axes] = 1
        else:
            for axis in self.axes:
                new_shape[axis] = 1
        
        return broadcast_to(reshape(out_grad, new_shape), lhs_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        if a.device != b.device:
          b = b.to(a.device)
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_grad = matmul(out_grad, transpose(rhs, (-1, -2))) 
        rhs_grad = matmul(transpose(lhs, (-1, -2)), out_grad) 
        
        for _ in range(len(lhs_grad.shape) - len(lhs.shape)):
          lhs_grad = lhs_grad.sum(0)
        for _ in range(len(rhs_grad.shape) - len(rhs.shape)):
          rhs_grad = rhs_grad.sum(0)

        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        return out_grad / lhs
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        node_data = node.realize_cached_data().numpy()
        node_data[node_data > 0] = 1
        val = Tensor(node_data, device=out_grad.device, dtype=out_grad.dtype)
        return out_grad * val
        ### END YOUR SOLUTION

def relu(a):
    return ReLU()(a)

class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        val = node.inputs[0]
        return out_grad * (init.ones(*out_grad.shape, device=out_grad.device,requires_grad=False) - power_scalar(tanh(val), 2))
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        shape = args[0].shape
        new_shape = shape[ :self.axis] + (len(args),) + shape[self.axis: ]
        out = array_api.empty(shape=new_shape, device=args[0].device)
        for i, v in enumerate(args):
            out[(slice(None),) * self.axis + (i,) + (slice(None),) * (len(shape) - self.axis)] = v
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # return list(split(out_grad, self.axis))
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        seqs = []
        for s in A.shape:
            seqs.append(slice(0, s))
        out = []
        for i in range(A.shape[self.axis]):
            seqs[self.axis] = slice(i, i+1)
            A_c = A[tuple(seqs)].compact()
            out.append(A_c.compact().reshape(new_shape))
        return tuple(out)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        alen = len(a.shape)
        if self.axes == None:
            axes = tuple(range(alen))
        else:
            axes = self.axes

        return a.flip(axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return  flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        idxs = []
        for i in a.shape: 
          idxs.append(slice(0, i))
        shape_list = list(a.shape)
        for axis in range(len(a.shape)):
          if axis in self.axes:
              shape_list[axis] = a.shape[axis] * (self.dilation+1)
              idxs[axis] = slice(0, shape_list[axis], self.dilation+1)
        ans = a.device.full(tuple(shape_list), 0)
        ans[tuple(idxs)] = a
        return ans
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        idxs = []
        for i in a.shape: 
          idxs.append(slice(0, i))
        shape_list = list(a.shape)
        for i in range(len(a.shape)):
            if i in self.axes:
                shape_list[i] = (a.shape[i] // (1 + self.dilation))
                idxs[i] = (slice(0, a.shape[i], (1 + self.dilation)))
        return a[tuple(idxs)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        if A.device != B.device:
          B = B.to(A.device)
        if self.padding > 0:
            A = A.pad(axes=((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))

        N, H, W, C_in = A.shape
        K1, K2, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        H_out, W_out = (H - K1) // self.stride + 1,  (W - K2) // self.stride + 1

        im_2_col = A.as_strided(shape=(N, H_out, W_out, K1, K2, C_in), strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)).compact()
        out = (im_2_col.compact().reshape((N * H_out * W_out, K1 * K2 * C_in)).compact() @ B.compact().reshape((K1 * K2 * C_in, C_out)).compact())
        out = out.compact().reshape((N, H_out, W_out, C_out))
        
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        out_grad_dilated = dilate(out_grad, axes=(1, 2), dilation=self.stride - 1)
        B_flip = flip(B, axes=(0, 1))
        out_grad_dilated_t = transpose(transpose(out_grad_dilated, (0, 1)), (1, 2))
        grad_a = conv(out_grad_dilated,transpose(B_flip, axes=(2, 3)),padding=B.shape[0]-1 - self.padding)
        grad_b = transpose(conv(transpose(A, (0, 3)),out_grad_dilated_t,padding=self.padding,),(0, 1),)
        return grad_a, transpose(grad_b, (1, 2))
        ### END YOUR SOLUTION



def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)

