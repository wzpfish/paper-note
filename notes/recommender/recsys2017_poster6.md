# The Importance of Song Context in Music Playlists

## 论文内容

### What

音乐推荐的一个任务是生成一个推荐列表(automated music playlist generation)。一些研究工作在生成列表中的下一首歌的时候会考虑之前生成的歌，即song context。但其实并不清楚song context能怎么影响生成歌曲的质量。

因此，这篇文章主要探讨的问题是：song context如何影响music playlist generation结果？

### How

为了得出上述问题的结论，作者用了三种方法来作对比。

1. Song Popularity. 该方法只考虑歌曲本身的流行度，并**没有用到任何context**。
2. Song-based CF. 该方法用歌曲是否出现在歌单中作为向量（类似bag-of-word）来计算歌曲间相似度。生成下一首歌的时**只找_context最后一首歌**最相似的歌曲。
3. RNN. 该方法类似nlp中的sequence generation，**把所有song-context都利用起来**。

另外，作者还用random打分作为对照。

### Data

AotM-2011 & 8tracks

数据过滤：

1. 歌单必须包含3个不同的歌手，每个歌手最多只有包含两首歌。防止是歌手专辑。
2. 歌单必须包含5首歌。
3. 每首歌必须至少出现在10个歌单中。保证歌曲数据够多，模型能学到东西。
4. 80%训练，20%测试。只出现在测试集的歌去掉。因为模型学不到任何东西。

### Result

<img src="/figures/recommender/recsys2017_poster6.png" alt="" width="700px" height="500px">

如图第一列所示，RNN和song popularity的效果居然差不多。。CF的效果比random略微好一点点。

为了探究为什么这样，作者把预测的下一首歌分为popular(前10%)和不popular。

如图第二列所示，song popularity预测popular歌曲的能力远远强于其他几种，而RNN预测不popular歌曲的能力是最强的。

由于popular歌曲数量占大多数，song popularity的总体效果很不错，但是对长尾歌曲预测就不行了。而RNN的鲁棒性是最强的，对于长尾歌曲和popular歌曲的预测能力很稳定。

### Conclusion

Song context如何影响music playlist generation结果？

1. popular歌曲造成的影响掩盖了song context对歌曲生成效果的影响。
2. 考虑context的歌曲生成对于**长尾歌曲生成**非常有效。


## 借鉴意义

* 在推荐的时候，可以考虑popular-based + model-based结合。

原文: [The Importance of Song Context in Music Playlists](http://ceur-ws.org/Vol-1905/recsys2017_poster6.pdf)



