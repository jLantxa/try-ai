"""
    Copyright 2019 Javier Lancha Vázquez

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import keras

import matplotlib.pyplot as plt
import numpy as np

import datetime, time
import os

def load_mnist_data():
    # Import MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Reshape dataset (60000, 784) -> (-1, 28, 28, 1)
    x_train_rshape = x_train.reshape([-1, 28, 28, 1])
    x_test_rshape = x_test.reshape([-1, 28, 28, 1])

    # Convert images from [0, 255] to [-1, 1]
    x_train_rshape = 2.0*(x_train_rshape/255)-1
    x_test_rshape = 2.0*(x_test_rshape/255)-1
    # Relabel to categorical
    y_train = keras.utils.np_utils.to_categorical(y_train)
    y_test = keras.utils.np_utils.to_categorical(y_test)
    return (x_train_rshape, y_train), (x_test, y_test)

class SequentialModel():
    def __init__(self, name, load_weights=False):
        self.name = name
        self.model = keras.models.Sequential()
        self.build()
        if load_weights:
            self.load_weights()

    def build(self):
        pass

    def compile(self, optimizer):
        pass

    # Save model and weights
    def save_weights(self):
        self.model.save_weights(self.name + ".h5")
        print("Saved model weights", self.name, "to file")

    # Load model and weights
    def load_weights(self):
        try:
            self.model.load_weights(self.name + ".h5")
            print("Loaded model weights from file")
        except Exception as ex:
            print(ex)
            print("Could not load model", self.name, "from file. Building a blank model.")


class Discriminator(SequentialModel):
    def __init__(self, name, img_shape, load_weights=False):
        self.img_shape = img_shape
        SequentialModel.__init__(self, name, load_weights)

    # TODO: Tweak configuration
    def build(self):
        self.model.add(keras.layers.Flatten(input_shape=self.img_shape))
        self.model.add(keras.layers.Dense(512))
        self.model.add(keras.layers.LeakyReLU(alpha=0.2))
        self.model.add(keras.layers.Dense(256))
        self.model.add(keras.layers.LeakyReLU(alpha=0.2))
        self.model.add(keras.layers.Dense(1, activation='sigmoid'))
        self.model.summary()

    def compile(self, optimizer):
        self.model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])


class Generator(SequentialModel):
    def __init__(self, name, img_shape, noise_size, load_weights=False):
        self.img_shape = img_shape
        self.noise_size = noise_size
        SequentialModel.__init__(self, name, load_weights)

    # TODO: Tweak configuration
    def build(self):
        self.model.add(keras.layers.Dense(256, input_dim=self.noise_size))
        self.model.add(keras.layers.LeakyReLU(alpha=0.2))
        self.model.add(keras.layers.BatchNormalization(momentum=0.8))
        self.model.add(keras.layers.Dense(512))
        self.model.add(keras.layers.LeakyReLU(alpha=0.2))
        self.model.add(keras.layers.BatchNormalization(momentum=0.8))
        self.model.add(keras.layers.Dense(1024))
        self.model.add(keras.layers.LeakyReLU(alpha=0.2))
        self.model.add(keras.layers.BatchNormalization(momentum=0.8))
        self.model.add(keras.layers.Dense(np.prod(self.img_shape), activation='tanh'))
        self.model.add(keras.layers.Reshape(self.img_shape))
        self.model.summary()

    def compile(self, optimizer):
        self.model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])


class MNIST_GAN():
    def __init__(self, load_weights=False):
        self.img_shape = (28, 28, 1)
        self.noise_size = 100
        optimizer = keras.optimizers.Adam(1e-4, 0.5)

        self.discriminator = Discriminator("discriminator", self.img_shape, load_weights)
        self.discriminator.compile(optimizer)

        self.generator = Generator("generator", self.img_shape, self.noise_size, load_weights)

        self.adversarial = keras.models.Sequential()
        # Make discriminator non-trainable for the combined model
        self.discriminator.model.trainable = False
        self.adversarial.add(self.generator.model)
        self.adversarial.add(self.discriminator.model)
        self.adversarial.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, steps, batch_size=128, save_weights=False, snap_step=1000):
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
        session_dir = "gen/"+timestamp
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)

        (x_train, _), (_, _) = load_mnist_data()

        # Generate real/fake labels
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        self.plot(0, dir=session_dir)
        for step in range(1, steps+1):
            """ Fit discriminator """
            rand_ix = np.random.randint(0, x_train.shape[0], batch_size)
            noise = np.random.normal(0, 1, (batch_size, self.noise_size))
            real_imgs = x_train[rand_ix]
            fake_imgs = self.generator.model.predict(noise)

            """
            train_on_batch() returns a vector of metrics
                index 0 contains the loss
                index 1 contains the accuracy
            """
            d_real_metrics = self.discriminator.model.train_on_batch(real_imgs, real_labels)
            d_fake_metrics = self.discriminator.model.train_on_batch(fake_imgs, fake_labels)
            d_mean_metrics = 0.5 * np.add(d_real_metrics, d_fake_metrics)

            """ Fit generator """
            noise = np.random.normal(0, 1, (batch_size, self.noise_size))
            g_metrics = self.adversarial.train_on_batch(noise, real_labels)
            print("%d: [D loss: %.3f - accuracy: %.3f] [G loss: %.3f - accuracy: %.3f]" \
                % (step, d_mean_metrics[0], d_mean_metrics[1], g_metrics[0], g_metrics[1]))
            if step % snap_step == 0:
                self.plot(step, dir=session_dir)
                print("Saved snapshot")
                if save_weights:
                    print("Saved weights")
                    self.discriminator.save_weights()
                    self.generator.save_weights()

    def plot(self, step, rows=5, cols=5, dir=None):
        noise = np.random.normal(0, 1, (rows * cols, self.noise_size))
        generated = self.generator.model.predict(noise)
        generated = 0.5*(generated+1)

        figure, axis = plt.subplots(rows, cols)
        for i in range(rows):
            for j in range(cols):
                axis[i,j].imshow(generated[i*cols+j, :, :, 0], cmap='gray')
                axis[i,j].axis('off')
        if dir == None:
            figure.savefig("gen/%d.png" % step, dpi=120)
        else:
            figure.savefig(dir+"/%d.png" % step, dpi=120)
        plt.close()
