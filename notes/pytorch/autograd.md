# autograd 理解
pytorch 中的 `torch.autograd` 是一个计算 vector-jacobian product 的引擎。

那么，为什么是计算这个呢？

<img src="/figures/pytorch/autograd.jpg" alt="" width="700px" height="700px">

通过推导可以发现，神经网络中关于权重求导，都可以通过这个计算框架得到，因此 vector-jacobian product engine 是一个很通用的计算引擎。

## Reference
* [AUTOGRAD: AUTOMATIC DIFFERENTIATION](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)