# Distributed Representations of Sentences and Documents

## PV-DM 公式推导
<img src="/figures/nlp/d2v_pv_dm.jpg" alt="" width="700px" height="800px">

## PV-DBOW 公式推导
这个推导和 PV-DM 大同小异，就不写了。

## inference
doc2vec 的 inference 过程比较特殊，它在 inference 的时候也需要 train。

具体地，在 inference 的时候，我们的目标是得到没有见过的 doc 的 doc vector。因此，需要把 document 的 lookup table 拓展一维给它用，
然后用新加入的 document 来训练自己的 doc 向量。训练过程中，之前得到的词向量以及 hidden layer->output layer 的权重是不变的，只更新新加入
的 document 对应的 doc vec，直到收敛。收敛后的向量就是新 document 的向量。

根据推导的直观解释（见推导图），一个句子的向量可以看作输出层词向量的加权平均。可以想像假如两个句子只是词的顺序不一样，由于输出层词向量是固定不变的，唯一变化的就是输出层的误差，相当于就是调整输出层词向量的权重。

## 论文中提到的一些 tricks
* PV-DM 通常都比 PV-DBOW 好，且几乎达到两个向量拼在一起用的效果。
* context 大小选 5-12 比较合适，最好用交叉验证试试。
* 实验中，词和 doc 向量都选了400维；context 大小为8；当句子长度小于9时，会 pad 一个 null 词；标点符号当作一个正常词

## Reference
* 原文 [Distributed Representations of Sentences and Documents](https://arxiv.org/pdf/1405.4053.pdf) 