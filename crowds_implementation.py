# General imports
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import numpy as np
import tensorflow as tf

# Function to set the initial weights of the layer
# The size for Friends dataset is: 5x5x3
# the initial weights are the same for all annotators - these are identity matrices
def initialize_weights(shape, dtype=None):
    out = np.zeros(shape)
    for a in range(shape[2]):
        for c in range(shape[0]):
            out[c,c,a] = 1.0
    return out

# Defining the layer class
class CrowdsClassification(Layer):
    # Initialize the layer class
    def __init__(self, output_dim, num_annotators, connection_type='MW', **kwargs):
        self.output_dim = output_dim
        self.num_annotators = num_annotators
        self.connection_type = connection_type
        super().__init__(**kwargs)

    # Implement the 'build' method
    # 'build' is the main method and its only purpose is to build the layer properly
    def build(self, input_shape):
        self.kernel = self.add_weight("CrowdLayer", shape=(self.output_dim, self.output_dim, self.num_annotators),
                        initializer=initialize_weights, trainable=True)
        # Call the base class, build the method:
        super(CrowdsClassification, self).build(input_shape)

        return self.kernel

    # Define the 'call' method
    # 'call' does the exact working of the layer during training process
    def call(self, input_data):
        return K.dot(input_data, self.kernel)

    # Define 'compute_output_shape' function
    # It copmutes the output shape using the input shape and the output dimension set while initializing the layer
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim, self.num_annotators)

# Define the loss function used to optimize the model
# Masking is used for missing labels    
class MaskedMultiCrossEntropy(object):
    def loss(self, y_true, y_pred):
        vec = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred, axis=1)
        mask = tf.equal(y_true[:,0,:],-1)
        zer = tf.zeros_like(vec)

        loss = tf.where(mask, x=zer, y=vec)

        return loss