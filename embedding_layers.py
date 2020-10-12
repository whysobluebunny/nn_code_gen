from layer import Layer


# Turns positive integers (indexes) into dense vectors of fixed size.

# input_dim: Integer. Size of the vocabulary, i.e. maximum integer index + 1.
# output_dim: Integer. Dimension of the dense embedding.
# embeddings_initializer: Initializer for the embeddings matrix (see keras.initializers).
# embeddings_regularizer: Regularizer function applied to the embeddings matrix (see keras.regularizers).
# embeddings_constraint: Constraint function applied to the embeddings matrix (see keras.constraints).
# mask_zero: Boolean, whether or not the input value 0 is a special "padding" value that should be masked out.
# input_length: Length of input sequences, when it is constant.

# keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform',
# embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None,
# mask_zero=False, input_length=None)
class Embedding(Layer):
    def __init__(self, input_dim=None, output_dim=None, embeddings_initializer=None,
                 embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None,
                 mask_zero=None, input_length=None):
        Layer.__init__(self)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_initializer = embeddings_initializer
        self.embedding_regularizer = embeddings_regularizer
        self.activity_regularizer = activity_regularizer
        self.embedding_constraint = embeddings_constraint
        self.mask_zero = mask_zero
        self.input_length = input_length
        self.name = 'Embedding'

    def set_input_dim(self, value):
        self.input_dim = value

    def set_output_dim(self, value):
        self.output_dim = value

    def set_embedding_initializer(self, value):
        self.embedding_initializer = value

    def set_activity_regularizer(self, value):
        self.activity_regularizer(self, value)

    def set_embedding_regularizer(self, value):
        self.embedding_regularizer = value

    def set_embedding_constraint(self, value):
        self.embedding_constraint = value

    def set_mask_zero(self, value):
        self.mask_zero = value

    def set_input_length(self, value):
        self.input_length = value
