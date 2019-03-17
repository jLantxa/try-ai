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

"""
    This is the "Hello World!" of Convolutional Neural Networks
    In this example we will build a CNN using Keras to be able to classify
    entries from the Fashion MNIST dataset.
    The trained model can be saved into a json for further training or as a
    final classifier model.
"""

import keras

class FashionMNIST:
    def __init__(self):
        self.model = keras.models.Sequential()
        self.build()
        self.load_weights()

    # Build the model by adding layers
    def build(self):
        self.model.add(keras.layers.Conv2D(32, (5, 5), padding="same", input_shape=[28, 28, 1]))
        self.model.add(keras.layers.MaxPool2D(2, 2))
        self.model.add(keras.layers.Conv2D(64, (5, 5), padding="same", input_shape=[14, 14, 1]))
        self.model.add(keras.layers.MaxPool2D(2, 2))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(1024, activation="relu"))
        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(10, activation="softmax"))
        self.compile()

    # Compile the model using ADAM as optimiser
    def compile(self):
        self.model.compile(keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, x_train, y_train, epochs=10):
        self.model.fit(x_train, y_train, validation_split=0.10, batch_size=64, epochs=epochs)

    def validate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    # Save model and weights
    def save_weights(self):
        self.model.save_weights("fmnist.h5")
        print("Saved model to json")

    # Load model and weights
    def load_weights(self):
        try:
            self.model.load_weights("fmnist.h5")
            self.compile()
            print("Loaded model from json")
        except Exception as ex:
            print(ex)
            print("Could not load model from file. Building a blank model.")

if __name__ == "__main__":
    fmnist = FashionMNIST()
    fmnist.model.summary()

    # Import fashion_mnist dataset
    # If you are curious, try this same model with the original MNIST dataset
    # by loading keras.datasets.mnist instead. You should be getting an accuracy
    # of over 99% by tweaking the model a little.
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Reshape dataset (60000, 784) -> (-1, 28, 28, 1)
    # The original shape of the dataset is fit for MLPs that have a flattened
    # input layer but our CNN takes input images of 28x28
    x_train_rshape = x_train.reshape([-1, 28, 28, 1])
    x_train_rshape = x_train_rshape/255
    x_test_rshape = x_test.reshape([-1, 28, 28, 1])
    x_test_rshape = x_test_rshape/255

    # Relabel to categorical
    y_train = keras.utils.np_utils.to_categorical(y_train)
    y_test = keras.utils.np_utils.to_categorical(y_test)

    fmnist.train(x_train_rshape, y_train, epochs=10)
    loss, accuracy = fmnist.validate(x_test_rshape, y_test)
    print("Test loss:", loss, "Test accurracy:", accuracy)

    fmnist.save_weights()
