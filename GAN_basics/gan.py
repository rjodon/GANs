"""
Brief
-----
    This is a generic implemetation of Generative Adversial Network (GAN).
    
Details
-------
    See more and refs are givi: https://en.wikipedia.org/wiki/Generative_adversarial_network

Note
----
    Don't forget to configure plaidml and select the device you want to use to run your model on (CPU, GPU...)
    It can be achieved using the command plaidml-setup in a terminal with your virtual environnement activated.
      
    - compare you different devices, it is not guaranteed that your GPU beats the CPU.
    - plaidml support experimental drivers, though results are likely to be not correct.

"""
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import subprocess, shlex

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np


class GAN:
    def __init__(self, x_train, n_samples, latente_dim):
        """
        GANManStyle
        
        Parameters
        ----------
        x_train np.array
            the dataset to train
        n_samples int
            the number of samples in the dataset
        latente_dim int
            the size of the latent vector
        """
        self.x_train = x_train
        self.n_samples = n_samples
        self.latent_dim = latente_dim

        self.x_train_shape = (self.x_train.shape[1], self.x_train.shape[2])
        self.sample_count = 0

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.build_generator()

        # The generator takes noise as input and generates an output
        noise_in = Input(shape=(self.latent_dim,))
        generator_output = self.generator(noise_in)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated outputs as input and checks validity
        validity = self.discriminator(generator_output)

        # The combined model
        # Trains the generator to fool the discriminator
        self.combined = Model(noise_in, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        """ Build the generator """

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.x_train_shape), activation='tanh'))
        model.add(Reshape(self.x_train_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        generator_output = model(noise)

        return Model(noise, generator_output)

    def build_discriminator(self):
        """ Build the discrimator """

        model = Sequential()

        model.add(Flatten(input_shape=self.x_train_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        r = Input(shape=self.x_train_shape)
        validity = model(r)

        return Model(r, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        """
        Trains the GAN
        
        Parameters
        ----------
        epochs int
        batch_size int
        sample_interval int

        """
        # Normalise values between -1 and 1
        self.x_train = self.x_train / 127.5 - 1.
        #self.x_train = np.expand_dims(self.x_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        # Noise
        self.noise = np.random.normal(0, 1, (25, self.latent_dim))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of data
            idx = np.random.randint(0, self.n_samples, batch_size)
            x_sample = self.x_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(x_sample, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("{: d} [D loss: {:05.4f}, acc.: {:05.2f} %] [G loss: {:05.4f}]".format(epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images()

    def sample_images(self):
        row, col = 5, 5
        gen_imgs = self.generator.predict(self.noise)
        self.path_to_images="images"
        
        # Rescaling
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(row, col)
        cnt = 0
        for i in range(row):
            for j in range(col):
                axs[i,j].imshow(gen_imgs[cnt, :,:], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("{}/{:05d}.png".format(self.path_to_images, self.sample_count))
        plt.close()
        self.sample_count += 1
    
    def make_video(self):
        """ Lazy implementation... """
        cmd = os.path.join(os.getcwd(), "generate_anim.sh")
        cmd = shlex.split(cmd, posix=True)
        res = subprocess.check_output(cmd)
        

if __name__ == '__main__':
    
    EPOCHS=8000
    BATCH_SIZE=64
    SAMPLE_INTERVAL=200
   
    # Load the dataset
   
    (x_train, _), (_, _) = mnist.load_data()
    n_samples = x_train.shape[0]
   
    # Define and run the GAN training
   
    gan = GAN(x_train=x_train, n_samples=n_samples, latente_dim=100)
    gan.train(epochs=EPOCHS, batch_size=BATCH_SIZE, sample_interval=SAMPLE_INTERVAL)
    gan.make_video()