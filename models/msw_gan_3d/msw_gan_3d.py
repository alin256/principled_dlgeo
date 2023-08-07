import tensorflow as tf
from tensorflow import keras
from tensorflow import image

tf.config.run_functions_eagerly(True)


def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    # print("LOSSES: f{0}, r{1}".format(real_loss, fake_loss))
    return fake_loss - real_loss


def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


class WMSGANSN3D(keras.Model):
    """
    WGAN2D (Wasserstein Generative Adversarial Network) model with Gradient Penalty Training
    """

    def __init__(
            self,
            discriminator,
            generator,
            latent_shape,
            discriminator_extra_steps=3,
            generator_extra_steps=1,
            gp_weight=10.0,
            real_image_resize_method=image.ResizeMethod.BILINEAR
    ):
        super(WMSGANSN3D, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_shape = latent_shape
        self.d_steps = discriminator_extra_steps
        self.g_steps = generator_extra_steps
        self.gp_weight = gp_weight
        self.real_image_resize_method = real_image_resize_method

    def compile(self, d_optimizer, g_optimizer, d_loss_fn=discriminator_loss, g_loss_fn=generator_loss):
        super(WMSGANSN3D, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    @tf.function
    def call(self, x):
        batch_size = tf.shape(x)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, *self.latent_shape))
        generated_images = self.generator(random_latent_vectors, training=True)
        return generated_images

    @tf.function
    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        x_1 = tf.keras.layers.AveragePooling3D(pool_size=(8, 8, 8))(real_images)
        x_2 = tf.keras.layers.AveragePooling3D(pool_size=(4, 4, 4))(real_images)
        x_3 = tf.keras.layers.AveragePooling3D(pool_size=(2, 2, 2))(real_images)
        x_high_res = tf.keras.layers.AveragePooling3D(pool_size=(1, 1, 1))(real_images)

        real_images = [x_1, x_2, x_3, x_high_res]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(shape=(batch_size, *self.latent_shape))

            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)
                d_loss = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the discriminator loss using the fake and real image logits
                d_loss = d_loss + 0.0001 * tf.reduce_mean(tf.square(tf.concat([fake_logits, real_logits], axis=0)))

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        for i in range(self.g_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, *self.latent_shape))
            with tf.GradientTape() as tape:
                # Generate fake images using the generator
                generated_images = self.generator(random_latent_vectors, training=True)
                # Get the discriminator logits for fake images
                gen_img_logits = self.discriminator(generated_images, training=True)
                # Calculate the generator loss
                g_loss = self.g_loss_fn(gen_img_logits)

            # Get the gradients w.r.t the generator loss
            gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
            # Update the weights of the generator using the generator optimizer
            self.g_optimizer.apply_gradients(
                zip(gen_gradient, self.generator.trainable_variables)
            )

        return {"d_loss": d_loss, "g_loss": g_loss}
