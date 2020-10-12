# Main Layer class which is an ancestor of others

class Layer:
    def __init__(self):
        self.name = None

    def set_input(self, shape):
        self.input_shape = shape

    def set_output(self, shape):
        self.output_shape = shape

    def wrap(self, st):
        return f'\'{st}\''

    # Converts layer parameter list to the script which is going to be used to construct a model
    def build_str(self):
        str_r = f'keras.layers.{self.name}( '
        for a in self.__dict__.items():
            if a[1] is not None and a[0] != 'name':
                str_r += f'{str(a[0])}={str(a[1]) if not isinstance(a[1], str) else self.wrap(str(a[1]))},'
        str_r = str_r[0:-1] + ')'
        return str_r

    def __str__(self):
        return self.build_str()
