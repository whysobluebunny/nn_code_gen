from layer import Layer


class Pooling(Layer):
    def __init__(self, pool_size=None, strides=None, padding='valid', data_format='channels_last'):
        Layer.__init__(self)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format

    def set_pool_size(self, value):
        self.pool_size = value

    def set_strides(self, value):
        self.strides = value

    def set_padding(self, value):
        self.padding = value

    def set_data_format(self, value):
        self.data_format = value


# keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')
class MaxPooling1D(Pooling):
    def __init__(self, pool_size=None, strides=None, padding=None, data_format=None):
        Pooling.__init__(self, pool_size, strides, padding, data_format)
        self.name = 'MaxPooling1D'


# keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
class MaxPooling2D(Pooling):
    def __init__(self, pool_size=None, strides=None, padding=None, data_format=None):
        Pooling.__init__(self, pool_size, strides, padding, data_format)
        self.name = 'MaxPooling2D'


# keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
class MaxPooling3D(Pooling):
    def __init__(self, pool_size=None, strides=None, padding=None, data_format=None):
        Pooling.__init__(self, pool_size, strides, padding, data_format)
        self.name = 'MaxPooling3D'


# keras.layers.AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')
class AveragePooling1D(Pooling):
    def __init__(self, pool_size=None, strides=None, padding=None, data_format=None):
        Pooling.__init__(self, pool_size, strides, padding, data_format)
        self.name = 'AveragePooling1D'


# keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
class AveragePooling2D(Pooling):
    def __init__(self, pool_size=None, strides=None, padding=None, data_format=None):
        Pooling.__init__(self, pool_size, strides, padding, data_format)
        self.name = 'AveragePooling2D'


# keras.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
class AveragePooling3D(Pooling):
    def __init__(self, pool_size=None, strides=None, padding=None, data_format=None):
        Pooling.__init__(self, pool_size, strides, padding, data_format)
        self.name = 'AveragePooling3D'
