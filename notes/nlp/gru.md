# GRU 理解
gru 是2014年 Bengio 提出的一种解决 long-term dependency 问题的 rnn 结构。

传统的rnn在计算梯度的时候需要根据序列长度层层求导，导数累计到前面就容易出现「梯度消散」或「梯度爆炸」的问题。为了解决这个问题，就有人提出了
LSTM 和 GRU 等结构。本文解释为什么 GRU 能够解决这个问题。

## 图解
如下图：
<img src="/figures/nlp/gru.jpg" alt="" width="700px" height="800px">

## 直观理解
观察 Zt 这个门，可以发现，如果模型认为当前的 unit 不需要保存与当前输入有关的信息时，可以将 Zt 设为1，这时 `h(t) = h(t-1)`，这里有点类似于 self-residual 的机制，让梯度可以顺利传回到上一层。从 nlp 的任务直观上来解释，这就是抛弃当前的输入词（比如一些无用词），而完全使用历史信息传递给下一个 step。而 rt 这个门用于控制历史信息对当前词的影响。如果历史信息对当前词的信息没有用，那就用当前词 xt 表示新的意思；否则，就协同表示新的意思。

## Reference
* [Learning Phrase Representations using RNN Encoder–Decoder
for Statistical Machine Translation](https://www.aclweb.org/anthology/D14-1179)
* [三次简化一张图：一招理解LSTM/GRU门控机制](https://zhuanlan.zhihu.com/p/28297161)