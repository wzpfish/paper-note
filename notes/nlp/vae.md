# VAE 解释

variational autoencoder 是一种 autoencoder。它和一般的 autoencoder 不太一样，感觉是 variational inference 的一个副产物。

以手写数字识别为例，我们如果要生成手写数字 x，需要一个隐变量 z 来指导生成，比如数字的方向，笔画宽度之类的。如果我们知道怎么得到隐变量 z，以及 p(x|z)，那么就可以先通过采样 z，然后通过 p(x|z) 得到生成的 x。

那么如何根据一个 x，找到它的隐变量 z，即如何求 p(z|x) 呢？

根据贝叶斯理论，我们知道 p(z|x) = p(x|z) * p(z) / p(x)。这个式子中，我们可以假设 p(x|z) 服从某个分布，p(z) 服从某个分布。但是真实数据的分布 p(x) 我们是不知道的，因此 p(z|x) 没法求。怎么办？

我们可以构造一个 q(z|x)，使得它服从某个有解析形式的分布，如正太分布。然后使 q(z|x) 尽可能的贴近 p(z|x) 这个分布。怎么衡量两个分布是否贴近？用 KL 散度。那么问题转换为，找到一个已知形式的分布 q(z|x)，使得 KL(q(z|x)||p(z|x)) 尽可能小。我们把这个 KL 散度写出来，可以得到：

<img src="/figures/nlp/vae1.jpg" alt="" width="700px" height="800px">
<img src="/figures/nlp/vae2.jpg" alt="" width="700px" height="800px">

上面的推导假设 p(x|z) 服从正太分布，如果服从伯努利分布会怎么样呢？就是 decoder 预测的不是均值和方差，只是一个均值罢了，然后 loss 把伯努利分布的概率公式填进去即可。

## generative
在生成阶段，如何生成新的 samples 呢？我们只需要 sample 一个 z ~ N(0, I)，然后喂给 decoder 即可。这里我的理解是由于我们在 train 的时候，使得 q(z|x) 尽可能贴近 N(0, I) 了，所以从 N(0, I) 中 sample 出来的变量也可以很好地还原出样本。

## tensorflow 实现

[代码在这里](/codes/vae)

## 总结
感觉这个 VAE 真的是很巧妙的结合了变分推断和神经网络，值得细细品味。。

## Reference
* [VAE viedo course](https://www.youtube.com/watch?v=uaaqyVS9-rM)
* [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)
* [Tutorial on Variational Autoencoders](https://arxiv.org/pdf/1606.05908.pdf)
