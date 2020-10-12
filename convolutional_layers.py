from layer import Layer
from kb_support import KernelBiasSupport


# keras.layers.Conv1D(filters, kernel_size, strides=1,
# padding='valid', data_format='channels_last',
# dilation_rate=1, activation=None, use_bias=True,
# kernel_initializer='glorot_uniform',
# bias_initializer='zeros', kernel_regularizer=None,
# bias_regularizer=None, activity_regularizer=None,
# kernel_constraint=None, bias_constraint=None)

# filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
# kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
# strides: An integer or tuple/list of a single integer, specifying the stride length of the convolution.
# Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
# padding: One of "valid", "same" or "causal" (case-insensitive). "valid" means no padding. "same"
# results in padding evenly to the left/right or up/down of the input such that output has the same height/width
# dimension as the input. "causal" results in causal (dilated) convolutions, e.g. output[t] does not depend on
# input[t+1:]. Useful when modeling temporal data where the model should not violate the temporal order.
# See WaveNet: A Generative Model for Raw Audio, section 2.1.
# data_format: A string, one of channels_last (default) or channels_first.
# dilation_rate: an integer or tuple/list of a single integer, specifying the dilation rate to use for dilated
# convolution.
# Currently, specifying any dilation_rate value != 1 is incompatible with specifying any strides value != 1.
# groups: A positive integer specifying the number of groups in which the input is split along the channel axis.
# Each group is convolved separately with filters / groups filters. The output is the concatenation of all the
# groups results along the channel axis. Input channels and filters must both be divisible by groups.
# activation: Activation function to use. If you don't specify anything, no activation is applied
# ( see keras.activations).
# use_bias: Boolean, whether the layer uses a bias vector.
# kernel_initializer: Initializer for the kernel weights matrix ( see keras.initializers).
# bias_initializer: Initializer for the bias vector ( see keras.initializers).
# kernel_regularizer: Regularizer function applied to the kernel weights matrix (see keras.regularizers).
# bias_regularizer: Regularizer function applied to the bias vector ( see keras.regularizers).
# activity_regularizer: Regularizer function applied to the output of the layer (its "activation")
# ( see keras.regularizers).
# kernel_constraint: Constraint function applied to the kernel matrix ( see keras.constraints).
# bias_constraint: Constraint function applied to the bias vector ( see keras.constraints).
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


# 1D convolution layer (e.g. temporal convolution).
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


# 2D convolution layer (e.g. spatial convolution over images).
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


# 3D convolution layer (e.g. spatial convolution over volumes).
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
