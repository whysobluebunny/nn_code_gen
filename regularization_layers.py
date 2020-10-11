from layer import Layer


# tf.keras.layers.Dropout(rate, noise_shape=None, seed=None, **kwargs)
class Dropout(Layer):
    def __init__(self, rate=0.1, noise_shape=None, seed=None):
        Layer.__init__(self)
        self.name = "Dropout"
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed

    def set_rate(self, value):
        self.rate = value

    def set_noise_shape(self, value):
        self.noise_shape = value

    def set_seed(self, value):
        self.seed = value