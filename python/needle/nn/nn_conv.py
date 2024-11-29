"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(
                self.kernel_size**2 * self.in_channels,
                self.kernel_size**2 * self.out_channels,
                shape=(
                    self.kernel_size,
                    self.kernel_size,
                    self.in_channels,
                    self.out_channels,
                ),
            ),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        bound = 1 / (self.in_channels * self.kernel_size**2) ** 0.5
        if bias:
            self.bias = Parameter(
                init.rand(
                    self.out_channels,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:  # x: NCHW
        ### BEGIN YOUR SOLUTION
        nchw_x = ops.transpose(ops.transpose(x, (1, 2)), (2, 3))
        nchw_out = ops.conv(
            nchw_x, self.weight, stride=self.stride, padding=(self.kernel_size) // 2
        )
        if self.bias:
            nchw_out += self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(
                nchw_out.shape
            )
        nhwc_out = ops.transpose(ops.transpose(nchw_out, (2, 3)), (1, 2))
        return nhwc_out
        ### END YOUR SOLUTION
