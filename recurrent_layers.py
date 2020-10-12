from layer import Layer
from kb_support import KernelBiasSupport


# Abstract base class for recurrent layers.
class Recurrent(Layer, KernelBiasSupport):
    def __init__(self, units, activation, recurrent_activation,
                 use_bias, kernel_initializer, recurrent_initializer,
                 bias_initializer, kernel_regularizer,
                 recurrent_regularizer, bias_regularizer, activity_regularizer,
                 kernel_constraint, recurrent_constraint, bias_constraint, dropout,
                 recurrent_dropout, implementation, return_sequences, return_state,
                 go_backwards, stateful, unroll):
        Layer.__init__(self)
        KernelBiasSupport.__init__(self, use_bias, kernel_initializer, bias_initializer, kernel_regularizer,
                                   bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint)
        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.recurrent_initializer = recurrent_initializer
        self.recurrent_regularizer = recurrent_regularizer
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.implementation = implementation
        self.return_sequencies = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.recurrent_constraint = recurrent_constraint

    def set_units(self, value):
        self.units = value

    def set_activation(self, value):
        self.activation = value

    def set_recurrent_activation(self, value):
        self.recurrent_activation = value

    def set_recurrent_initializer(self, value):
        self.recurrent_initializer = value

    def set_recurrent_reguralizer(self, value):
        self.recurrent_regularizer = value

    def set_dropout(self, value):
        self.dropout = value

    def set_recurrent_dropout(self, value):
        self.recurrent_dropout = value

    def set_implementation(self, value):
        self.implementation = value

    def set_return_sequencies(self, value):
        self.return_sequencies = value

    def set_return_state(self, value):
        self.return_state = value

    def set_go_backwards(self, value):
        self.go_backwards = value

    def set_stateful(self, value):
        self.stateful = value

    def set_unroll(self, value):
        self.unroll = value

    def set_recurrent_constraint(self, value):
        self.recurrent_constraint = value


# keras.layers.LSTM(units, activation='tanh', recurrent_activation='sigmoid',
# use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
# bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None,
# recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
# kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0,
# recurrent_dropout=0.0, implementation=2, return_sequences=False, return_state=False,
# go_backwards=False, stateful=False, unroll=False)
class LSTM(Recurrent):
    def __init__(self, units=None, activation=None, recurrent_activation=None,
                 use_bias=None, kernel_initializer=None, recurrent_initializer=None,
                 bias_initializer=None, unit_forget_bias=None, kernel_regularizer=None,
                 recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=None,
                 recurrent_dropout=None, implementation=None, return_sequences=None, return_state=None,
                 go_backwards=None, stateful=None, unroll=None):
        Recurrent.__init__(self, units, activation, recurrent_activation,
                           use_bias, kernel_initializer, recurrent_initializer,
                           bias_initializer, kernel_regularizer,
                           recurrent_regularizer, bias_regularizer, activity_regularizer,
                           kernel_constraint, recurrent_constraint, bias_constraint, dropout,
                           recurrent_dropout, implementation, return_sequences, return_state,
                           go_backwards, stateful, unroll)
        self.unit_forget_bias = unit_forget_bias
        self.name = 'LSTM'


# Gated Recurrent Unit
# units, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform',
# recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None,
# bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
# dropout=0.0, recurrent_dropout=0.0, implementation=2, return_sequences=False, return_state=False, go_backwards=False,
# stateful=False, unroll=False, reset_after=False
class GRU(Recurrent):
    def __init__(self, units=None, activation=None, recurrent_activation=None, use_bias=None,
                 kernel_initializer=None,
                 recurrent_initializer=None, bias_initializer=None, kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=None, recurrent_dropout=None, implementation=None, return_sequences=None, return_state=None,
                 go_backwards=None,
                 stateful=None, unroll=None, reset_after=None):
        Recurrent.__init__(self, units, activation, recurrent_activation,
                           use_bias, kernel_initializer, recurrent_initializer,
                           bias_initializer, kernel_regularizer,
                           recurrent_regularizer, bias_regularizer, activity_regularizer,
                           kernel_constraint, recurrent_constraint, bias_constraint, dropout,
                           recurrent_dropout, implementation, return_sequences, return_state,
                           go_backwards, stateful, unroll)
        self.reset_after = reset_after
        self.name = 'GRU'
