# Automatic Article Commenting: the Task and Dataset

这篇 paper 主要贡献是公开了一个大规模的新闻评论数据集，以及衡量评论质量的几个改进 metrics。

## Dataset
作者从腾讯新闻中爬取了2017.07 - 2017.08时间段内的新闻和评论数据，并且过滤了长度小于30词的新闻以及长度小于20词的评论。

最终得到了如下规模的数据集：

|      | train  | dev   | test |
| ---- | ----   | ----  | ---- |
| #Articles | 191502 | 5000 | 1610 |
| #Cmts/Articles  | 27 | 27 | 27 |
| #Upvotes/Cmt| 5.9 | 4.9 | 3.4 |

测试集中的每条评论都给两个 labeler 标注过。

## Metrics
由于每条评论的打分不一样，作者认为用 BLEU,METEOR 等 reference-based metrics 不能很好地表现出这一点。因此提出了对应的 weighted metrics，包括：weighted BLEU, Weighted METEOR, Weighted ROUGE, Weighted CIDEr。这些其实都是给 reference 加上分数权重。

作者做了实验，计算 candidate 的人工打分与 metrics 打分的 Spearman 和 Pearson 相关性，发现 weighted 的 metrics 比原来的 metrics 和人工打分的一致性更强。

## Others

除了上面两个主要的贡献之外，作者做了一些评论匹配与生成的实验，有如下现象：
1. 仅用新闻 tile 做匹配/生成没有 title+content 效果好。
2. 生成的效果没有匹配的效果好。
3. 生成时，有 attention 比没有 attention 效果好。

当然，这些结果只是作者做了一些很简单的实验，并没有探索 SOA 模型。

## Reference
* [Automatic Article Commenting: the Task and Dataset](https://aclweb.org/anthology/P18-2025)