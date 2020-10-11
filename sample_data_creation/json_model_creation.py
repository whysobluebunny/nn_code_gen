import neural_network
import convolutional_layers
import pooling_layers
import core_layers
import regularization_layers
import reshaping_layers

import jsonpickle

nn = neural_network.NeuralNetwork()
nn.add_layer(core_layers.Input(shape=(28, 28, 1)))
nn.add_layer(convolutional_layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
nn.add_layer(pooling_layers.MaxPooling2D(pool_size=(2, 2)))
nn.add_layer(convolutional_layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
nn.add_layer(pooling_layers.MaxPooling2D(pool_size=(2, 2)))
# layers.Flatten(),
nn.add_layer(reshaping_layers.Flatten())
nn.add_layer(regularization_layers.Dropout(0.5))
# layers.Dropout(0.5),
nn.add_layer(core_layers.Dense(10, activation="softmax"))

nn.set_compile(neural_network.CompileParams(loss="categorical_crossentropy", optimizer="adam"))
nn.set_fit(neural_network.FitParams(batch_size=128, epochs=15, validation_split=0.1))

nnJ = jsonpickle.encode(nn)
with open("..\\generated\\model.json", "w+") as f:
    f.write(nnJ)