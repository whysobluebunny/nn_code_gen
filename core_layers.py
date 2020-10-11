from layer import Layer
from kb_support import KernelBiasSupport


# keras.layers.Dense(units, activation=None, use_bias=True,
# kernel_initializer='glorot_uniform', bias_initializer='zeros',
# kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
# kernel_constraint=None, bias_constraint=None)
class Dense(Layer, KernelBiasSupport):
    def __init__(self, units=None, activation=None, use_bias=True, kernel_initializer=None,
                 bias_initializer=None, kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
        Layer.__init__(self)
        KernelBiasSupport.__init__(self, use_bias, kernel_initializer, bias_initializer, kernel_regularizer,
                                   bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint)
        self.units = units
        self.activation = activation
        self.name = 'Dense'

    def set_units(self, value):
        self.units = value

    def set_activation(self, value):
        self.activation = value


# keras.layers.Activation(activation)
class Activation(Layer):
    def __init__(self):
        Layer.__init__(self)
        self.activation = None
        self.name = 'Activation'

    def set_activation(self, value):
        self.activation = value

# keras.layers.Dropout(rate, noise_shape=None, seed=None)
class Dropout(Layer):
    def __init__(self, rate=None, noise_shape=None, seed=None):
        Layer.__init__(self)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.name = 'Dropout'

    def set_rate(self, value):
        self.rate = value

    def set_noise_shape(self, value):
        self.noise_shape = value

    def set_seed(self, value):
        self.seed = value


# keras.layers.Flatten(data_format=None)
class Flatten(Layer):
    def __init__(self, data_format=None):
        Layer.__init__(self)
        self.data_format = data_format
        self.name = 'Flatten'

    def set_data_format(self, value):
        self.data_format = value

# keras.engine.input_layer.Input() - вранье какое-то
class Input(Layer):
    def __init__(self, shape=None, batch_shape=None, dtype=None, sparse=None, tensor=None):
        Layer.__init__(self)
        self.shape = shape
        self.batch_shape = batch_shape
        self.dtype = dtype
        self.sparse = sparse
        self.tensor = tensor
        self.name = 'Input'

    def set_shape(self, value):
        self.shape = value

    def set_batch_shape(self, value):
        self.batch_shape = value

    def set_dtype(self, value):
        self.dtype = value

    def set_sparse(self, value):
        self.sparse = value

    def set_tensor(self, value):
        self.tensor = value