# 各种 attention 机制对比

## 1 Introduction
在 NMT 任务中，通常的框架是有个 rnn-based encoder 来对平行语料的输入编码，编码为一个 dense 的向量。然后利用该向量以及一个 rnn-based decoder 来预测输出词。这个框架相当于一个判别模型，直接预测 p(y|x)。

在 decoder 的过程中，如何利用 encoder 的 hidden layer 就比较多样了。最简单的就是直接将 encoder 最后一层的 hidden 看作整个句子的表示，并作为 decoder 的 initial hidden。只用 encoder 最后一层的信息会不会无法准确表示一个完整的语义？是不是有些词翻译根本不需要整个句子的语义，而是部分词的语义就行？这时，attention model 就出场了。

attention model 的基本思路是，在 decoder 每次预测下一个输出词的概率时，不只用上一层的 hidden，还要用上 encoder 各层的 hidden 信息。怎么用上呢？最直观的就是各层 hidden 加权求和。权重怎么来呢？通过计算 decoder 当前 hidden 与 encoder 的每一层 hidden 的 "相似度"或者"对齐度"，通常就是做个线性变换映射到同一个空间之后做点积。最后，为了归一化权重，加一个 softmax 即可。

## 2 我所接触过的不同 attention
具体应该如何求 encoder 的 hidden 的加权向量 c，如何用上这个 c，不同的 attention 机制有不同的做法。主要注意如何计算 attention 的权重以及如何使用 attention 向量 c。

### 2.1 Bahdanau's attention
  
Bahdanau's attention 将 attention 作为 rnn 的额外输入，具体结构如下图所示：
<img src="/figures/nlp/attention_bahdanau.jpg" alt="" width="700px" height="800px">

- 计算 attention 权重：用 decoder 上一层的 hidden 以及 encoder 每一层的 hidden 做一系列线性变换得到权重
- 用 attention 向量：将 attention 向量 c 作为额外的 input 输入到 decoder 的神经单元中。在具体实现时，可以将 c 和 decoder 的 input 拼接在一起做线性变换。

### 2.2 Luong's attention
Luong 提出了一种通用 attention 框架，以及两种不同的 attention 计算方法，包括 global attention 以及 local attention。

Luong的 attention 框架如下图所示：
<img src="/figures/nlp/attention_luong_framework.jpg" alt="" width="700px" height="400px">

可以发现，在这种框架中，attention 向量 c 不是作为 decoder 当前神经元的额外输入，而是直接和当前层的 hidden concat 在一起，做个线性变换和激活，得到新的表示。相当于是「后处理」，而 Bahdanau's attention 相当于是「先处理」。

global 和 local attention 的框架一样，区别是计算 attention 向量 c 时候，attention 权重的计算方式。

#### 2.2.1 Global Attention
Global attention的计算方式比较直观：
<img src="/figures/nlp/attention_luong_global.jpg" alt="" width="700px" height="400px">

global 的意思就是利用所有 encoder 的 hidden，不同的计算方式只是 encoder 的 hidden 和 decoder 的 hidden 怎么算一个分出来。其中，location 比较特殊，它不用任何 encoder 的 hidden，仅仅根据 decoder 的 hidden 来求 attention weights。

#### 2.2.2 Local Attention
global attention的缺点是，当平行语料的 input 很长，比如是篇长文章时，其实很多信息是没有必要用的，只需要关注局部的信息就行了。因此，local attention 被提出用来解决这个问题。

local attention 的思想是找一个对齐位置 pt，每一次只利用该位置及其两边的 D 个词，即[pt - D, pt + D]。这里的 D 是一个超参数，可以根据经验来选。

local attention 也分为两种，如下图：
<img src="/figures/nlp/attention_luong_local.jpg" alt="" width="700px" height="500px">

看了公式 local attention 的原理非常明了。

## 3 总结
|| Bahdanau's attention | Luong's global attention | Luong's local attention
---- | --- | --- | ---
attention weights 计算方式 | 只有一种 | general, concat 等多种| 同 global attetion
attention 向量计算方式 | 用 encoder 的所有 hidden | 用 encoder 的所有 hidden | 用 encoder 的部分 hidden
attetion 向量使用方式 | 与 decoder 当前层的 input 拼接在一起，作为神经元的输入 | 得到 decoder 当前层的 hidden 之后，与 attention 向量拼接在一起，来预测输出词的概率。| 同 global attention 

## 4 Reference
* [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)
* [Effective Approaches to Attention-based Neural Machine Translation](http://aclweb.org/anthology/D15-1166)
* [Practical PyTorch: Translation with a Sequence to Sequence Network and Attention](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb)