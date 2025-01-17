"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None

class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentumomentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad is None:
                continue
            val = param.grad.data
            if param not in self.u:
                self.u[param] = np.zeros_like(param.realize_cached_data())
            val = val + self.weight_decay * param.realize_cached_data()
            self.u[param] = self.momentumomentum * self.u[param] + (1-self.momentumomentum)*val
            param.data =param.data- self.lr *self.u[param]
        ### END YOUR SOLUTION

class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.momentum = {}
        self.velocity = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            if param.grad is None:
                continue
            # val = param.grad.realize_cached_data().astype(param.dtype)
            val = param.grad.data
            if param not in self.momentum:
                self.momentum[param] =np.zeros_like(param.realize_cached_data())
                self.velocity[param] = np.zeros_like(param.realize_cached_data())
            val = val + self.weight_decay * param.realize_cached_data()
            self.momentum[param] =(self.beta1 * self.momentum[param]) + ((1 - self.beta1) * val)
            self.velocity[param] = (self.beta2 * self.velocity[param]) +((1 - self.beta2) * (val ** 2))
            u_hat = self.momentum[param] /(1 - self.beta1 ** self.t)
            v_hat = self.velocity[param]/ (1 - self.beta2 ** self.t)
            param.data=param.data -self.lr * u_hat /((v_hat **0.5) + self.eps)
        ### END YOUR SOLUTION