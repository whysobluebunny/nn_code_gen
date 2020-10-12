# processed data to use in the script
class ShapedData:
    def __init__(self, x_training, y_training, x_test=None, y_test=None):
        self.x_training = x_training
        self.x_test = x_test
        self.y_test = y_test
        self.y_training = y_training

    def set_x_training(self, value):
        self.x_training = value

    def set_x_test(self, value):
        self.x_test = value

    def set_y_training(self, value):
        self.y_training = value

    def set_y_test(self, value):
        self.y_test = value
