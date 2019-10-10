import tensorflow as tf

class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim + latent_dim)
        ])

        self.decoder_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
          tf.keras.layers.Conv2DTranspose(
              filters=64,
              kernel_size=3,
              strides=(2, 2),
              padding="SAME",
              activation='relu'),
          tf.keras.layers.Conv2DTranspose(
              filters=32,
              kernel_size=3,
              strides=(2, 2),
              padding="SAME",
              activation='relu'),
          # No activation
          tf.keras.layers.Conv2DTranspose(
              filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
        ])
    
    def encode(self, x):
        mean, sigma = tf.split(self.encoder_net(x), num_or_size_splits=2, axis=1)
        return mean, sigma

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    
    def reparameterize(self, mean, sigma):
        eps = tf.random.normal(shape=mean.shape)
        return eps * sigma + mean

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)


optimizer = tf.keras.optimizers.Adam(1e-4)

def compute_loss(model, x):
    mean, sigma = model.encode(x)
    z = model.reparameterize(mean, sigma)
    x_logit = model.decode(z)

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    # [batch_size, 1]
    marginal_likelihood = -tf.reduce_sum(cross_entropy, axis=[1, 2, 3])
    neg_KL_divergence = 0.5 * tf.reduce_sum(
        1 + tf.math.log(1e-8 + tf.square(sigma)) - tf.math.square(mean) - tf.math.square(sigma),
        axis=1)
    ELBO = tf.reduce_mean(marginal_likelihood + neg_KL_divergence)
    loss = -ELBO
    return loss

def compute_appply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
