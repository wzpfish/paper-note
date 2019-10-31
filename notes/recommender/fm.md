# Factorization Machines
因子分解机是一个 SVM + 因子分解的结合体。它考虑了特征之间的 interaction，且特征间的交互权重是用低维向量的点积来表示的。

## Introduction
先来看一下 FM 要解决的问题定义：假设给定数据集 D，每一条样本为 (x, y)，x 是一个非常稀疏的特征集合，即 x 中非零特征远小于总特征数。我们需要一个模型 f，可以很好地 mapping x -> y，即使得 y = f(x)。

下面来看看 FM 模型：

<img src="/figures/recommender/fm.jpg" alt="" width="600px" height="800px">

## Summary
FM 有以下几点好处：
1. FM 模型和 SVM 相比，就是把交叉特征的权重矩阵 W(n x n) 分解成了 V(n x k)，用 V x V转置来代替权重矩阵。这样做最大的好处是能够好处理特征稀疏的情况。
2. FM 能同时用于离散值和连续值特征。
3. 计算复杂度低。
  
## Reference
* [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)