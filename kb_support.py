class KernelBiasSupport:
    def __init__(self, use_bias=None, kernel_initializer=None, bias_initializer=None, kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

    def set_use_bias(self, value):
        self.use_bias = value

    def set_kernel_initializer(self, value):
        self.kernel_initializer = value

    def set_bias_initializer(self, value):
        self.bias_initializer = value

    def set_kernel_regularizer(self, value):
        self.kernel_regulizer = value

    def set_bias_regularizer(self, value):
        self.bias_regulizer = value

    def set_activity_regularizer(self, value):
        self.activity_regulizer = value

    def set_kernel_constraint(self, value):
        self.kernel_constraint = value

    def set_bias_constraint(self, value):
        self.bias_constraint = value