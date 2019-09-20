# BLEU

BLEU(bilingual evaluation understudy) 是 IBM 于2002年提出的一种评估双语翻译质量的评价指标。该评价指标的基础思想是，**一个机器翻译的结果越接近专业人工翻译的结果越好**。这种评价标准是一种 reference-based metrics.

因此，该指标有两个重要组成部分：
1. reference: 即高质量的人工翻译语料。
2. closeness: 衡量机器翻译与人工翻译语料之间的距离的标准。


第一点 reference 没啥好说的，找人写或标就好了。主要是怎么衡量 closeness，BLEU 的思路是基于机器生成的句子中 n-gram 出现在标注数据中的比例，再加上一些修正手段。

## Modified n-gram precision

对于一个翻译任务，假设有 reference 数据集 {sourcei, referencei1, referencei2, ... referenceiN}, 以及 candidate 数据集 {sourcei, candidatei1, candidatei2 ... candidateiK} i=1,2 ... M。

即一共有 M 条评测数据，每条数据下面有 N 条人工写的翻译结果和 K 条机器翻译的结果。

modified n-gram precision 思想是，计算 n-gram 的精确率，只是该精确率是截断过的。具体来说，一般计算精确率是 (candidate 中，在 reference 里出现过的 ngram 个数)/(candidate 总 ngram 个数)。但是这样计算容易作弊，比如下面例子：

Candidate: the the the the the the the.

Reference 1: The cat is on the mat.

Reference 2: There is a cat on the mat.

precision = 7/7 = 1

因此，修正的 precision 分子会做一个截断，使其不超过单个 reference 中出现该 ngram 的最多次数，即 min(count, max_count_in_one_reference)

最终，ngram 的精确率写为：

<img src="/figures/mt/bleu/precision.png" alt="">

## brevity penalty
若果只用 modified-precision 还会存在一个问题：短句会占优势，比如只有一个字「的」，就很有可能取得1.0的 precision。

BLEU 的解决方法是加一个 brevity penalty。具体计算方法为，假设所有 candiate 的长度之和为 c, 且对于每个 candiate，从 reference 集合中找出一个与其长度最匹配的 reference，这些 reference 的长度和为 r，于是BP的计算公式为：

<img src="/figures/mt/bleu/bp.png" alt="">

## Geometric average of n-gram precision up to N

对于 1->N gram，计算他们的 modified n-gram precision 的加权几何平均数（通常权重取1/N），最后乘以长度惩罚BP：

<img src="/figures/mt/bleu/bleu.png" alt="">

## 总结

BLEU 主要就是算 precision，只是为了让这个 precision 更加合理，加入了：
1. modified n-gram precision
2. brevity penalty

## Reference
[BLEU: a Method for Automatic Evaluation of Machine Translation](https://www.aclweb.org/anthology/P02-1040)