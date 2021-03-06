from layer import Layer


# Softmax activation function.
# keras.layers.Softmax(axis=-1)
# axis: Integer, or list of Integers, axis along which the softmax normalization is applied.
class Softmax(Layer):
    def __init__(self, axis=None):
        Layer.__init__(self)
        self.axis = axis
        self.name = 'Softmax'

    def set_axis(self, value):
        self.axis = value


# Rectified Linear Unit activation function.
# keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
# max_value: Float >= 0. Maximum activation value. Default to None, which means unlimited.
# negative_slope: Float >= 0. Negative slope coefficient. Default to 0.
# threshold: Float. Threshold value for thresholded activation. Default to 0.
class ReLU(Layer):
    def __init__(self, max_value=None, negative_slope=None, threshold=None):
        Layer.__init__(self)
        self.max_value = max_value
        self.negative_slope = negative_slope
        self.threshold = threshold
        self.name = 'ReLU'

    def set_max_value(self, value):
        self.max_value = value

    def set_negative_slope(self, value):
        self.negative_slope = value

    def set_threshold(self, value):
        self.threshold = value
