from layer import Layer


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