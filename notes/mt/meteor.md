# METEOR

METEOR(Metric for Evaluation of Translation with Explicit ORdering) 是2005年 CMU 提出的一种衡量机器翻译好坏的评估标准。

该评估标准旨在解决 BLEU 存在的几个问题：
1. 没有考虑 recall
2. 没有考虑同义词、同词根词等之间的匹配
3. 缺少显示的翻译对齐信息。比如一些常用词，虽然翻译一样，但其实不不一定就是原文里词的翻译。
4. N-gram 的几何平均导致一个 n-gram precision 是0就全是0了。

因此，METEOR 的解决方法是：
1. 找出显示的词对齐信息。
2. 度量 candidate 与 reference 的 closeness 的时候加上 recall 的考虑，且认为 recall 比 precision 更重要。

## 计算方法

METEOR 的计算步骤比较复杂，主要分为两步：
1. 找出词对齐信息（candidate 中的一个词对齐到 reference 中的0或1个词）。
2. 计算 closeness score。

### 词对齐
在计算词对齐信息的时候又分为多个 stages，每个 stage 分为两个 phase：
* phase1: 找出所有对齐的 candidates。
* phase2: 根据这些 candidates 找出最大的对齐。

之所以存在多个 stage，是因为在找对齐 candidates 的时候可以根据不同的策略找，比如 exact(词完全一样), Porter stemmer(词根一样)，WN synonymy(同义词)。不同的 stage 对应不同的策略。后面的 stage 服从前面的，因此默认的 stage 顺序是 exact -> Porter stemmer -> WN synonymy。

### Closeness 度量
有了对齐信息后，就可以计算 closeness score：

Score = Fmean * (1 - Penalty)

其中 Fmean 为调和平均： (10PR)/(R+9P)，P 为 precision，R 为 recall，可见 recall 比 precision 重要得多。

Penalty 是对对齐的惩罚，定义如下：

<img src="/figures/mt/meteor/penalty.png" alt="">

其中 #unigrams_matched 表示对齐上的 unigram 个数，#chunks 表示连续对齐（在 candiate 中连续，且在 reference 中连续）的块数。 对齐的 chunks 数越多，表示不连续的翻译越多，惩罚越大；对齐上的 unigram 数越多，惩罚越小。

下面是 chunk 示例：

<img src="/figures/mt/meteor/chunk.png" alt="">

## 总结
METEOR 主要贡献有几个：
1. 考虑了同义词、同根词的匹配
2. 考虑了 F-Score，且认为 recall 比 precision 重要。
3. 考虑了翻译的语序信息，鼓励连续翻译词越多越好。

## Reference
[METEOR: An Automatic Metric for MT Evaluation with
Improved Correlation with Human Judgments](https://www.cs.cmu.edu/~alavie/papers/BanerjeeLavie2005-final.pdf)