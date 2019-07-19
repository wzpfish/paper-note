# Transformer 结构理解
Transformer 是 attention is all you need 中提出的一个翻译模型结构。该结构的特点是不用 rnn 和 cnn，而只用 attention 对输入句子进行编码；
以及只用 attention 做 decoder。 这么做的好处是加速训练以及方便模型并行。

这里主要讲一下模型结构。下图是论文中给出的模型结构：

<img src="/figures/nlp/transformer.png" alt="" width="700px" height="800px">

# Layers 介绍
下面从输入到输出经过的 layer 一个一个来说。

## Positional Encoding
在对句子进行 encoding 的时候，由于没有用到 rnn 和 cnn，我们需要一个方法可以保留词的位置信息，因此需要用 positional encoding 来记录这个信息。
positional encoding 其实就是对每个位置进行编码，编码的方式有两种：
1. 每个位置学习一个位置 embedding，维度和词的 embedding 维度相同。
2. 手动设置一个位置函数，该函数的输出与词的位置以及 embedding 的每一维度的位置有关（这样就能让模型利用这个额外信息）。

transformer 中用的是第二种，且论文中指出，用第一种方式和第二种方式得到的结果相差无几：

为什么用这个函数？我的理解是该函数满足几个特性：
* 输出区间为有穷区间[-1, 1]，比较容易控制
* PE(pos+k) 与 PE(pos) 有线性关系，方便模型学到不同距离词的相对位置信息。
* 周期函数，即只区分一个周期内的位置信息，周期够长就可以区分很大的位置区间。

## Multi-head Attention
名字听起来比较高级，其实就是在算 attention 的时候，把向量映射到多个不同的向量空间来计算多个 attention weights，从而得到多个 attention，最后将多个 attention 拼接在一起。

具体地，计算 attention 可以由如下框架表示：query, key, value => attention。
即通过 query 和 key 计算得到一个 attention weights，最终输出 value 关于这个 attention weights 的加权求和。例如在经典的 rnn 机器翻译模型中，query 就是 encoder 当前层的 hidden layer，key 和 value 是 encoder 的所有 hidden layer。

这么来看 Multi-head Attention 就很直观了：为了得到多个 attention 结果，必须将 query, key, value 映射到多个不同的向量空间中，再计算出多个 attention 的结果。

在 transformer 的 encoder 中，query key value 都是同一个矩阵，即上一层的 output 矩阵。

## Add & Norm
这里就是一个 residual + layer normalization。residual 可以解决梯度消失的问题，直接连个 input，导数就不会太小。ln 可以解决经过多层后，hidden 的分布有偏差的问题，将其重新归一化到一个近似独立同分布的空间中。

## Position-wise Feed Forward
即对每一个位置的向量做两次线性变换加激活。不同位置的线性变换矩阵是相同的，因此叫 position-wise。

position-wise 的线性变换从实现来说，就是 input 矩阵乘一个变换矩阵即可。 

# encoder 和 decoder
了解了每个 layer 的结构后，整个模型的结构就比较清晰了。

## encoder
encoder 是由6个 layer stack 在一起的，其中每个 layer 又包含了 multi-head attention 和 feed forward.

## decoder
decoder 也是由6个 layer stack 一起，其中每个 layer 和 encoder 的比，多了一层 multi-head attention，这层 attention 利用了 encoder 的结果。即 query 是 decoder 上一层的输出，key 和 value 是 encoder 的最终输出。其他的基本上一样。

# Transformer 实现
由于实现比较复杂，另写了个 notebook 记录了一下：[transormer 实现](https://nbviewer.jupyter.org/github/wzpfish/paper-note/blob/master/notes/nlp/transformer.ipynb)

# 总结
这里只是 high-level 的介绍了各个模型结构，具体公式细节以及模型实现，可以细细品味。我觉得这个文章最值得学习的点有几个：
1. positional encoding 来弥补非 rnn cnn 结构丢失位置信息的缺点，是不是普通的全连接网络也可以加入这个信息。
2. attetion 的 query，key，value 框架，非常有助于理解 attention 的设计。
3. self-attention 来学习句子之间词的关系

# Reference
* [attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
* [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
* [fairseq](https://github.com/pytorch/fairseq)