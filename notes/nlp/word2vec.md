# Word2vec 手撸

## CBOW
<img src="/figures/nlp/w2v_cbow.jpg" alt="" width="700px" height="800px">
<img src="/figures/nlp/w2v_cbow2.jpg" alt="" width="700px" height="500px">

## Skip-Gram
<img src="/figures/nlp/w2v_skipgram.jpg" alt="" width="700px" height="800px">

## Hierarchical Softmax
<img src="/figures/nlp/w2v_hierarchical.jpg" alt="" width="700px" height="800px">

层级softmax最终的 loss 只能用极大似然而不能用 cross entropy。因为如果用 cross entropy，每个叶子节点都会对 loss 有贡献，
那每个叶子节点的路径都需要更新，就等于更新整棵树了，没有起到加速的所用。

## Negative Sampling
<img src="/figures/nlp/w2v_negative_sample.jpg" alt="" width="700px" height="600px">

## Reference
* 原文 [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)
* [word2vec Parameter Learning Explained](https://arxiv.org/pdf/1411.2738.pdf)
* [The Matrix Cookbook](http://www.math.uwaterloo.ca/~hwolkowi//matrixcookbook.pdf)
