# Noise-Contrastive Estimation
NCE 是 Michael U. Gutmann （这哥们好像是 GoodFellow 的好朋友） 于2010年提出的参数估计方法。所谓参数估计，就是我们假设样本来源于某个参数模型，给定观测样本，估计样本服从的概率分布。

NCE 的核心思想是学习一个逻辑回归来判别观测数据和人工构造的噪声，拟合这个模型就可以预估观测数据分布的参数。同时，这个方法直接适用于非归一化的分布，即概率密度函数的积分不为1的。

由于这个方法是将真实样本与噪声进行对比，所以叫 Noise-Contrastive Estimation.

## NCE 解决什么问题？
上面说过，NCE 是一个参数概率密度估计的方法，那它是针对什么样的概率密度模型进行参数估计？看下面说明：

![](/figures/candidate_sampling/nce_problem.jpg)

一句话来说，NCE 是为了解决非归一化模型中，归一化因子 intractable 且计算代价非常大的模型的参数估计问题。

## NCE 的解决方法
上面提到，NCE 提出了一个 well-defined 目标函数，来解决上述问题，那这个 well-defined 目标函数是怎么来的呢？

NCE 通过将一个无监督的概率密度估计问题转换为有监督的逻辑回归问题，通过优化逻辑回归的目标函数，来估计概率密度的参数。具体的做法如下：

![](/figures/candidate_sampling/nce_method.jpg)

## NCE 目标函数定义

实际上，在 NCE 目标函数定义中，就是对 logistic regression 的似然函数除以 Td, 即：

![](/figures/candidate_sampling/nce_objective.png)

之所以不直接用逻辑回归的目标函数，我猜是因为方便后续的证明。

论文中证明，当 NCE 目标函数取得最大值时，Pm = Pd。也就是说，优化目标函数，我们就得到了 Pd 的概率密度估计。同时，这个 Pm 是天然归一化好的。

## Noise 分布与 Noise 个数的选择
文中给了以下几个建议：
1. ln(Pn) 是有解析表达式。
2. noise 很容易采样。
3. noise 分布尽可能接近真实数据分布，即 Pd。
4. noise 样本越大越好。

## NCE Loss 图解
在神经网络中，NCE loss 这么应用：

![](/figures/candidate_sampling/nce_loss.jpg)

因此，我感觉在神经网络中，NCE Loss 和 Softmax loss 其实没啥关系，它们是完全不同的两种估计 p(y|x) 的方法。理论上来说，这里的 logits 拟合完 log(p(y|x)) 的时，应该已经归一化好了，即 exp(logits) 就是概率 p(y|x) 了，不用做 softmax 去归一化。

## TensorFlow 中的实现
### Noise 分布选择
在 TensorFlow 的实现中，noise 分布选取的是一个 approximately log-uniform，即：

P(class) = (log(class + 2) - log(class + 1)) / log(total_classes + 1)

也就是说，它假设真实 label 分布中，label id 越小，出现概率越大。（考虑 NLP 中的词典，就可以用这个分布）

### Sigmoid 函数
在 NCE 论文中，逻辑回归函数应该是 1/(1+v*exp(-u))，但是在 TensorFlow 实现中直接用的 Sigmoid 函数。这里不知道有多大影响，还是我理解错了。。

## Reference
* [Noise-Contrastive Estimation of Unnormalized Statistical Models, with Applications to Natural Image Statistics](http://www.jmlr.org/papers/volume13/gutmann12a/gutmann12a.pdf)
* [tf.nn.nce_loss](https://www.tensorflow.org/api_docs/python/tf/nn/nce_loss)