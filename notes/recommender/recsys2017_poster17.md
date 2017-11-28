# Towards Effective Exploration/Exploitation in Sequential Music Recommendation

## 论文内容

### What

在音乐流推荐系统中，不仅要根据用户喜好推荐用户喜欢的歌曲（exploit），还需要给用户推荐一些新颖的歌曲（explore），以防音乐流被限制在一个很窄的方向中。那么问题来了，explore会对用户行为造成什么影响呢？

本文探究在给定之前音乐流（包括广告）context的情况下，explore会带来的影响。

### How

作者对比在给定之前的三个播放事件（包括音乐和广告），当前推送的是explore或exploit会对用户行为产生什么影响。例如给定Ad, Song, Song三个事件，是当前推送explore音乐，用户退出session的比例高，还是推送exploit音乐，用户退出session的比例高。这里广告顺序是随机生成的。

其实就是个简单的统计。

### Data

根据移动设备中的100万个session统计得到。

### Result

<img src="/figures/recommender/recsys2017_poster17.jpg" alt="" width="700px" height="500px">

如图所示，纵坐标表示explore音乐相比于exploit音乐，用户推出session增加的概率（x100）。可见无论context是什么，explore的音乐都更可能让用户退出。在Ad, Song, Ad的context下，甚至高了531.13%。但是在SAA的context下，高的不多。

### Conclusion

* explore音乐比exploit音乐容易让用户退出。
* session context会影响explore音乐让用户退出的可能性。

## 借鉴意义

在音乐推荐系统里面，explore的时机受session context的影响，所以应该根据实际情况尽可能选择最优的explore时机而不是盲目explore。

原文:[Towards Effective Exploration/Exploitation in Sequential Music Recommendation](http://ceur-ws.org/Vol-1905/recsys2017_poster17.pdf)
