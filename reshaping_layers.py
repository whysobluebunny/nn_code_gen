from layer import Layer

class Flatten(Layer):
    def __init__(self):
        Layer.__init__(self)
        self.name = "Flatten"