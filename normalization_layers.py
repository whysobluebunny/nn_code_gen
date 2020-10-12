from layer import Layer


# axis: Integer, the axis that should be normalized (typically the features axis). For instance, after a Conv2D
# layer with data_format="channels_first", set axis=1 in BatchNormalization.
# momentum: Momentum for the moving average.
# epsilon: Small float added to variance to avoid dividing by zero.
# center: If True, add offset of beta to normalized tensor. If False, beta is ignored.
# scale: If True, multiply by gamma. If False, gamma is not used. When the next layer is linear (also e.g. nn.relu),
# this can be disabled since the scaling will be done by the next layer.
# beta_initializer: Initializer for the beta weight.
# gamma_initializer: Initializer for the gamma weight.
# moving_mean_initializer: Initializer for the moving mean.
# moving_variance_initializer: Initializer for the moving variance.
# beta_regularizer: Optional regularizer for the beta weight.
# gamma_regularizer: Optional regularizer for the gamma weight.
# beta_constraint: Optional constraint for the beta weight.
# gamma_constraint: Optional constraint for the gamma weight.
# renorm: Whether to use Batch Renormalization. This adds extra variables during training.
# The inference is the same for either value of this parameter.
# renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to scalar
# Tensors used to clip the renorm correction.

# keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,
# center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
# moving_mean_initializer='zeros', moving_variance_initializer='ones',
# beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
class BatchNormalization(Layer):
    def __init__(self, axis=None, momentum=None, epsilon=None,
                 center=None, scale=None, beta_initializer=None, gamma_initializer=None,
                 moving_mean_initializer=None, moving_variance_initializer=None,
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None):
        Layer.__init__(self)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.moving_mean_initializer = moving_mean_initializer
        self.moving_variance_initializer = moving_variance_initializer
        self.beta_regularizer = beta_regularizer
        self.gamma_regularizer = gamma_regularizer
        self.beta_constraint = beta_constraint
        self.gamma_constraint = gamma_constraint
        self.name = 'BatchNormalization'

    def set_axis(self, value):
        self.axis = value

    def set_momentum(self, value):
        self.momentum = value

    def set_epsilon(self, value):
        self.epsilon = value

    def set_center(self, value):
        self.center = value

    def set_scale(self, value):
        self.scale = value

    def set_beta_initializer(self, value):
        self.beta_initializer = value

    def set_gamma_initializer(self, value):
        self.gamma_initializer = value

    def set_moving_mean_initializer(self, value):
        self.moving_mean_initializer = value

    def set_moving_variance_initializer(self, value):
        self.moving_variance_initializer = value

    def set_beta_regularizer(self, value):
        self.beta_regularizer = value

    def set_gamma_regularizer(self, value):
        self.gamma_regularizer = value

    def set_beta_constraint(self, value):
        self.beta_constraint = value

    def set_gamma_constraint(self, value):
        self.gamma_constraint = value
