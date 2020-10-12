import core_layers


# Wraps the string according to the needed format
def wrap(st):
    return f'\'{st}\''


# Custom type of exception
class NNException(Exception):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


# An optimizer is one of the two arguments required for compiling a Keras model
class Optimizer:
    def __init__(self, name, learning_rate):
        self.learning_rate = learning_rate
        self.name = name

    # converts optimizer parameters to the script
    def build_str(self):
        str_r = f'keras.optimizers.{self.name}( '
        for a in self.__dict__.items():
            if a[1] is not None and a[0] != 'name':
                str_r += f'{str(a[0])}={str(a[1]) if not isinstance(a[1], str) else wrap(str(a[1]))},'
        str_r = str_r[0:-1] + ')'

        return str_r

    def __str__(self):
        return self.build_str()


# Gradient descent (with momentum) optimizer.
class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        Optimizer.__init__(self, "SGD", learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov


# Optimizer that implements the Adam algorithm.
class Adam(Optimizer):
    def __init__(self, learning_rate=0.02, beta_1=0.9, beta_2=0.99, epsilon=1e-07):
        Optimizer.__init__(self, "Adam", learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon


# Optimizer that implements the RMSprop algorithm.
class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, rho=0.9, momentum=0.0, centered=False):
        Optimizer.__init__(self, "RMSprop", learning_rate)
        self.rho = rho
        self.momentum = momentum
        self.centered = centered


# Compilation parameters
class CompileParams:
    def __init__(self, optimizer='adam', loss='mean_squared_error'):
        self.optimizer = optimizer
        self.loss = loss

    # converts compiling parameters into script
    def build_str(self):
        str_r = ' '
        for a in self.__dict__.items():
            if a[1] is not None and a[0] != 'name':
                str_r += f'{str(a[0])}={str(a[1]) if not isinstance(a[1], str) else wrap(str(a[1]))},'
        str_r = str_r[0:-1]

        return str_r

    def __str__(self):
        return self.build_str()


# Fitting parameters
class FitParams:
    def __init__(self, object=None,
                 batch_size=32, epochs=200,
                 verbose=None,
                 callbacks=None, view_metrics=None, validation_split=None, validation_data=None,
                 shuffle=None, class_weight=None, sample_weight=None,
                 initial_epoch=None, steps_per_epoch=None, validation_steps=None):
        self.DEFAULT_VALUES = (32, 200)
        self.object = object
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.callbacks = callbacks
        self.view_metrics = view_metrics
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.shuffle = shuffle
        self.class_weight = class_weight
        self.sample_weight = sample_weight
        self.initial_epoch = initial_epoch
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

    def set_object(self, value):
        self.object = value

    def set_batch_size(self, value):
        self.batch_size = value

    def set_epochs(self, value):
        self.epochs = value

    def set_verbose(self, value):
        self.verbose = value

    def set_callbacks(self, value):
        self.callbacks = value

    def set_view_metrics(self, value):
        self.view_metrics = value

    def set_validation_split(self, value):
        self.validation_split = value

    def set_validation_data(self, value):
        self.validation_data = value

    def set_shuffle(self, value):
        self.shuffle = value

    def set_class_weight(self, value):
        self.class_weight = value

    def set_sample_weight(self, value):
        self.sample_weight = value

    def set_initial_epoch(self, value):
        self.initial_epoch = value

    def set_steps_per_epoch(self, value):
        self.steps_per_epoch = value

    def set_validation_steps(self, value):
        self.validation_steps = value

    def build_str(self):
        str_r = ' '
        for a in self.__dict__.items():
            if a[1] is not None and a[0] != 'name':
                if a[0] == 'callbacks':
                    str_r += f'{str(a[0])}={str(a[1])},'
                    continue
                if a[0] == 'DEFAULT_VALUES':
                    continue
                str_r += f'{str(a[0])}={str(a[1]) if not isinstance(a[1], str) else wrap(str(a[1]))},'
        str_r = str_r[0:-1]

        return str_r

    def __str__(self):
        return self.build_str()


class NeuralNetwork:
    def __init__(self, layers=[], compile_params=None, fit_params=None):
        self.layers = layers
        self.compile_params = compile_params
        self.fit_params = fit_params

    def add_layer(self, layer):
        self.layers.append(layer)

    def remove_layer(self, layer):
        self.layers.remove(layer)

    def set_compile(self, compile):
        self.compile_params = compile

    def set_fit(self, fit):
        self.fit_params = fit

    def build_body(self):
        if not self.layers:
            raise NNException('Json Config Error', 'Neural network cannot contain no layers. Please load another '
                                                   'configuration')
        body = ['model = keras.models.Sequential()']
        if not isinstance(self.layers[0], core_layers.Input):
            input_l = core_layers.Input(shape='input.shape')
            body.append(f'model.add({str(input_l)})')
        for l in self.layers:
            body.append(f'model.add({str(l)})')

        body.append(f'model.compile({str(self.compile_params)})')

        if self.fit_params.epochs == self.fit_params.DEFAULT_VALUES[1]:
            body.append('callback = keras.callbacks.EarlyStopping(monitor=\'loss\', min_delta=0.003, patience=2)')
            self.fit_params.set_callbacks('[callback]')

        body.append(f'model.fit( x=shaped_data.x_training,y=shaped_data.y_training,{str(self.fit_params)})')
        body.append(f'model.save(\'generated\\\\generated.h5\')')

        return body

    def __str__(self):
        tmp = ''
        for s in self.build_body():
            tmp += f'{s}\n'
        return tmp
