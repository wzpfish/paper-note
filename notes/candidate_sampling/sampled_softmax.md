# Sampled Softmax
Sampled softmax 是 Bengio 在2015年提出的，当时是为了解决这样一个问题：在 NMT 任务中，如果词表很大，那么训练的时候计算词表大小的 softmax logits 会非常耗时。于是他提出了一种训练时候的采样方法，可以提高训练效率。

## Introduction
关于 Sampled Softmax 的详细介绍与数学推导，直接看下图（基于 tensorflow 文档写的）：

<img src="/figures/candidate_sampling/sampled_softmax.jpg" alt="" width="800px" height="1200px">

## Implementation
抄了一份 TensorFlow 源码的实现，可以更加有助于对 Sampled Softmax 的理解。

下面代码其实执行了如下逻辑：
1. 对于一个 batch 的样本，采样 `num_sampled` 个相同的 labels。这里用的采样分布是 log uniform.
2. 只计算 `num_sampled + num_true` 个 labels 对应的 logits。即只挑出这些 labels 对应的 output embedding (`weights`) 和 output bias (`biases`) 来计算 logits。
3. 如果采样出来的 label 与真实 label 有重叠，则把采样出来的 label 对应的 logit 设为负无穷，这样 softmax 之后就不会有什么影响。
4. 对于每个 logit，减去 `Q(y|x)`，这里对应 log uniform 的概率。
5. 给真实 labels 一个均匀的分数，即 `1.0 / num_true`；给采样的 labels 一个0分。然后计算 softmax_cross_entropy。

```Python
def sampled_softmax_loss(weights, # [num_classes, dim]
                         biases, # [num_classes]
                         labels, # [batch_size, num_true]
                         inputs, # [batch_size, dim]
                         num_sampled, # int
                         num_classes, # int
                         num_true=1,
                         sampled_values=None,
                         subtract_log_q=True,
                         remove_accidental_hists=True):
    labels = tf.stop_gradient(labels, name="labels_stop_gradient")
    labels_flat = tf.reshape(labels, [-1]) # [batch_size * num_true]
    if sampled_values is None:
        # https://www.tensorflow.org/api_docs/python/tf/random/log_uniform_candidate_sampler
        sampled_values = tf.random.log_uniform_candidate_sampler(
            true_classes=labels,
            num_true=num_true,
            unique=True,
            num_sampled=num_sampled,
            range_max=num_classes)
    # stop gradients, 让梯度别回传。
    # sampled: [num_sampled]
    sampled, true_expected_count, sampled_expected_count = (
         tf.stop_gradient(s) for s in sampled_values)
    # sampled, true_expected_count, sampled_expected_count = sampled_values
        
    all_ids = tf.concat([labels_flat, sampled], axis=0) # [batch_size * num_true + num_sampled]
    
    all_w = tf.nn.embedding_lookup(weights, all_ids) # [batch_size * num_true + num_sampled, dim]
    true_w = tf.slice(all_w, [0, 0], [tf.shape(labels_flat)[0], -1]) # [batch_size * num_true, dim]
    sampled_w = tf.slice(all_w, [tf.shape(labels_flat)[0], 0], [-1, -1]) # [num_sampled, dim]
    sampled_logits = tf.matmul(inputs, sampled_w, transpose_b=True) # [batch_size, num_sampled]
    
    all_b = tf.nn.embedding_lookup(biases, all_ids) # [batch_size * num_true + num_sampled]
    true_b = tf.slice(all_b, [0], tf.shape(labels_flat)) # [batch_size * num_true]
    sampled_b = tf.slice(all_b, [tf.shape(labels_flat)[0]], [-1]) # [num_sampled]
    
    dim = tf.shape(true_w)[1]
    true_w = tf.reshape(true_w, [-1, num_true, dim]) # [batch_size, num_true, dim]
    true_logits = tf.matmul(tf.expand_dims(inputs, axis=1), true_w, transpose_b=True) # [batch_size, 1, num_true]
    true_logits = tf.reshape(true_logits, [-1, num_true]) # [batch_size, num_true]
    true_b = tf.reshape(true_b, [-1, num_true]) # [batch_size, num_true]
    true_logits += true_b
    sampled_logits += sampled_b # [batch_size, num_sampled]
    
    if remove_accidental_hists:
        # https://www.tensorflow.org/api_docs/python/tf/nn/compute_accidental_hits
        # [num_accidental_hits], [num_accidental_hits], [num_accidental_hits]
        acc_indices, acc_ids, acc_weights = tf.nn.compute_accidental_hits(labels, sampled, num_true=num_true)
        acc_indices = tf.cast(tf.expand_dims(acc_indices, axis=-1), dtype=tf.int64)
        acc_ids = tf.expand_dims(acc_ids, axis=-1)
        indices = tf.concat([acc_indices, acc_ids], axis=1) # [num_accidental_hits, 2]
        dense_shape = tf.cast(tf.shape(sampled_logits), dtype=tf.int64)
        sp = tf.sparse.SparseTensor(indices=indices, values=acc_weights, dense_shape=dense_shape)
        sampled_logits += tf.sparse.to_dense(sp, default_value=0, validate_indices=False)
        
    if subtract_log_q:
        true_logits -= tf.math.log(true_expected_count)
        sampled_logits -= tf.math.log(sampled_expected_count)
    
    out_logits = tf.concat([true_logits, sampled_logits], axis=1)
    out_labels = tf.concat([tf.ones_like(true_logits) / num_true, tf.zeros_like(sampled_logits)], axis=1)
    
    sampled_losses = tf.nn.softmax_cross_entropy_with_logits(labels=out_labels, logits=out_logits)
    return sampled_losses


num_classes = 1000
dim = 10
batch_size = 8
num_true = 1
num_sampled = 100

def create_labels():
    labels = np.random.uniform(0, 1, [batch_size, num_true]) * num_classes
    labels = labels.astype("int64")
    return tf.constant(labels, dtype=tf.int64)
                       
weights = tf.random.uniform([num_classes, dim], dtype=tf.float32)
biases = tf.random.uniform([num_classes], dtype=tf.float32)
labels = create_labels()
inputs = tf.random.uniform([batch_size, dim], dtype=tf.float32)
losses = sampled_softmax_loss(weights, biases, labels, inputs, num_sampled, num_classes)
print(losses)

losses = tf.nn.sampled_softmax_loss(weights, biases, labels, inputs, num_sampled, num_classes)
print(losses)
```

output:

```
tf.Tensor(
[4.061418  4.0273438 4.712978  3.5223796 4.8042254 2.2182093 5.95613
 4.8151927], shape=(8,), dtype=float32)
tf.Tensor(
[4.2707977 4.3034782 4.9677744 3.7459211 4.991434  2.5204499 6.1279364
 5.0813494], shape=(8,), dtype=float32)
```

以上输出表明代码抄对了，和 TensorFlow 库的实现输出差不多。。。

下面代码中，我计算了 loss 关于 weights 的导数，可以发现，只有108个 labels 对应的 output embedding 是会被更新的。每训练一个 batch，只更新 `batch_size * num_true + num_sampled` 个权重，可以大大节约训练成本。

```Python
with tf.GradientTape() as t:
    t.watch(weights)
    losses = sampled_softmax_loss(weights, biases, labels, inputs, num_sampled, num_classes)

print(t.gradient(losses, weights))
```

output:

```
IndexedSlices(indices=tf.Tensor(
[761 278 578 515 616 961 952 208   2  12  17 746   6  26 224 236   5 995
  21   0  31 122 299   4  30 156 793  19  20   1  29  34 795 137 200  35
   7  74  32 807 831 790 251 432  62  44 962 256  58   3  53  69  24  25
 676   8  52 139  79 197 125  83  18  82  61  11 446 212 238 438  88 312
 335  48  27 448 109 942 685 322  92  91  22 171  42  16 424  85 389 475
 328  86 108 176 311 162 393  71  15 237 970  77   9  57 161  33 583  37], shape=(108,), dtype=int64), values=tf.Tensor(
[[-0.6487777  -0.22576317 -0.5997946  ... -0.57982737 -0.1937098
  -0.7288522 ]
 [-0.7235955  -0.73820686 -0.01507875 ... -0.32494062 -0.62081456
  -0.9215719 ]
 [-0.8549423  -0.9025774  -0.65992934 ... -0.09531121 -0.8521775
  -0.8026252 ]
 ...
 [ 0.00750771  0.00738356  0.00510704 ...  0.00500872  0.00683329
   0.00771481]
 [ 0.11966164  0.12551966  0.08172677 ...  0.09716646  0.11517613
   0.12735564]
 [ 0.00725995  0.00825832  0.00596292 ...  0.00663924  0.00703381
   0.00813163]], shape=(108, 10), dtype=float32), dense_shape=tf.Tensor([1000   10], shape=(2,), dtype=int32))
```

## Reference
* [On Using Very Large Target Vocabulary for Neural Machine Translation](https://arxiv.org/pdf/1412.2007.pdf)
* [What is Candidate Sampling](https://www.tensorflow.org/extras/candidate_sampling.pdf)
* [tf.nn.sampled_softmax_loss](https://www.tensorflow.org/api_docs/python/tf/nn/sampled_softmax_loss)