from layer import Layer
from kb_support import KernelBiasSupport


# keras.layers.Conv1D(filters, kernel_size, strides=1,
# padding='valid', data_format='channels_last',
# dilation_rate=1, activation=None, use_bias=True,
# kernel_initializer='glorot_uniform',
# bias_initializer='zeros', kernel_regularizer=None,
# bias_regularizer=None, activity_regularizer=None,
# kernel_constraint=None, bias_constraint=None)
class Conv(Layer, KernelBiasSupport):
    def __init__(self, filters=None, kernel_size=None, strides=1, padding='valid',
                 data_format='channels_last', dilation_rate=1, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None):
        Layer.__init__(self)
        KernelBiasSupport.__init__(self, use_bias, kernel_initializer, bias_initializer, kernel_regularizer,
                                   bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint)
        self.strides = strides
        self.activation = activation
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate

    def set_strides(self, value):
        self.strides = value

    def set_activation(self, value):
        self.activation = value

    def set_filters(self, value):
        self.filters = value

    def set_kernel_size(self, value):
        self.kernel_size = value

    def set_padding(self, value):
        self.padding = value

    def set_data_format(self, value):
        self.data_format = value

    def set_dilation_rate(self, value):
        self.dilation_rate = value

class Conv1D(Conv):
    def __init__(self, filters=None, kernel_size=None, strides=1, padding='valid',
                 data_format='channels_last', dilation_rate=1, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None):
        Conv.__init__(self, filters, kernel_size, strides, padding,
                      data_format, dilation_rate, activation, use_bias, kernel_initializer,
                      bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer,
                      kernel_constraint, bias_constraint)
        self.name = 'Conv1D'


class Conv2D(Conv):
    def __init__(self, filters=None, kernel_size=None, strides=(1, 1), padding='valid', data_format=None,
                 dilation_rate=(1, 1),
                 activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
        Conv.__init__(self, filters, kernel_size, strides, padding,
                      data_format, dilation_rate, activation, use_bias, kernel_initializer,
                      bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer,
                      kernel_constraint, bias_constraint)
        self.name = 'Conv2D'


class Conv3D(Conv):
    def __init__(self, filters=None, kernel_size=None, strides=(1, 1, 1), padding='valid', data_format=None,
                 dilation_rate=(1, 1, 1),
                 activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
        Conv.__init__(self, filters, kernel_size, strides, padding,
                      data_format, dilation_rate, activation, use_bias, kernel_initializer,
                      bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer,
                      kernel_constraint, bias_constraint)
        self.name = 'Conv3D'
