from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K
import tensorflow as tf
from keras.engine.topology import Layer
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import Merge

from keras.layers.core import  Dense
from keras import objectives

#Gradient Reversal Layer from https://github.com/michetonu/gradient_reversal_keras_tf

def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y

class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'hp_lambda': self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class CustomRegularization(Layer):
    def __init__(self,output_dim, **kwargs):
        self.output_dim = output_dim
        super(CustomRegularization, self).__init__(**kwargs)
   
    def diff_loss(self,shared_feat, task_feat):
        task_feat -= K.mean(task_feat, 0)
        shared_feat -= K.mean(shared_feat, 0)
        task_feat = K.l2_normalize(task_feat, 1)
        shared_feat = K.l2_normalize(shared_feat, 1)
        
        correlation_matrix = K.batch_dot(K.permute_dimensions(task_feat, (0,2,1)), shared_feat, axes=[1,2])
        cost = K.mean(K.square(correlation_matrix)) #* 0.01
        cost = K.tf.where(K.tf.greater(cost,0), cost, 0, name='value')
        return cost

    def call(self,x ,mask=None):
        ld=x[0]
        rd=x[1]
        loss = self.diff_loss(ld, rd)
        self.add_loss(loss,x)
        #you can output whatever you need, just update output_shape adequately
        #But this is probably useful
        return ld

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2])

def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)


class WeightedCombination(Layer):
    """Layer that adds a list of inputs.
    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    ```
    """

    def __init__(self, **kwargs):
        super(WeightedCombination, self).__init__(**kwargs)
    
    def build(self, input_shape):
        alpha = np.full((len(input_shape)), 1./len(input_shape))
        self.weighing = K.variable(alpha)
        self.trainable_weights = [self.weighing]

    def call(self, inputs, mask=None):
        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('WeightedCombination must be called on a list of tensors '
                      '(at least 2). Got: ' + str(inputs))

        #output = inputs[0]
        output = tf.keras.backend.zeros_like(inputs[0])
        for i in range(0, len(inputs)):
          # output += inputs[i] #* self.weighing[i]
            output = K.tf.add(output, K.tf.multiply(inputs[i], self.weighing[i]))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

class WeightedCombinationLayer(Layer):
    """Layer that adds a list of inputs.
    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    ```
    """
    def __init__(self, **kwargs):
        #self.supports_masking = True
        super(WeightedCombinationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        alpha = np.full((len(input_shape),input_shape[0][1], input_shape[0][2]), 1./len(input_shape))
        self.weighing = K.variable(alpha)
        self.trainable_weights = [self.weighing]

    def call(self, inputs, mask=None):
        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('WeightedCombination must be called on a list of tensors '
                      '(at least 2). Got: ' + str(inputs))

        #output = inputs[0]
        output = tf.keras.backend.zeros_like(inputs[0])
        for i in range(0, len(inputs)):
          #output += inputs[i] * self.weighing[i]
          output = K.tf.add(output, K.tf.multiply(inputs[i], self.weighing[i][:][:]))
          #self.weighing = K.softmax(self.weighing)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

class WeightedCombinationLayerUnb(Layer):
    """Layer that adds a list of inputs.
    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    ```
    """
    def __init__(self, **kwargs):
        #self.supports_masking = True
        super(WeightedCombinationLayerUnb, self).__init__(**kwargs)

    def build(self, input_shape):
        #unbalanced init
        alpha = np.full((len(input_shape),input_shape[0][1], input_shape[0][2]), 1./len(input_shape))
        self.weighing = K.variable(alpha)
        self.trainable_weights = [self.weighing]

    def call(self, inputs, mask=None):
        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('WeightedCombination must be called on a list of tensors '
                      '(at least 2). Got: ' + str(inputs))

        output = inputs[0]
        #output = tf.keras.backend.zeros_like(inputs[0])
        for i in range(1, len(inputs)):
          #output += inputs[i] * self.weighing[i]
       #   output = K.tf.add(output, K.tf.multiply(inputs[i], self.weighing[i][:][:]))
          output = K.tf.multiply(output, K.sigmoid(K.tf.multiply(inputs[i], self.weighing[i][:][:])))
          #self.weighing = K.softmax(self.weighing)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

class CrossStitch(Layer):
    """Layer that adds a list of inputs.
    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    ```
    """
    def __init__(self, **kwargs):
        #self.supports_masking = True
        super(CrossStitch, self).__init__(**kwargs)

    def build(self, input_shape):
        alpha = np.full((len(input_shape),input_shape[0][1], input_shape[0][2]), 1./len(input_shape))
        self.weighing = K.variable(alpha)
        self.trainable_weights = [self.weighing]

    def call(self, inputs, mask=None):
        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('WeightedCombination must be called on a list of tensors '
                      '(at least 2). Got: ' + str(inputs))

        #output = inputs[0]
        output = tf.keras.backend.zeros_like(inputs[0])
        for i in range(0, len(inputs)):
          #output += inputs[i] * self.weighing[i]
          output = K.tf.add(output, K.tf.multiply(inputs[i], self.weighing[i][:][:]))
          #self.weighing = K.softmax(self.weighing)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape


