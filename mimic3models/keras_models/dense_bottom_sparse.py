from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Masking, Dropout, Flatten
from tensorflow.keras.layers import Bidirectional, TimeDistributed
from mimic3models.keras_utils import LastTimestep
from mimic3models.keras_utils import ExtendMask
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import Regularizer
import numpy as np

class L21(Regularizer):
    """Regularizer for L21 regularization.
    # Arguments
        C: Float; L21 regularization factor.
    """

    def __init__(self, C=0.):
        self.C = K.cast_to_floatx(C)

    def __call__(self, x):
        const_coeff = np.sqrt(K.int_shape(x)[1])
        return self.C*const_coeff*K.sum(K.sqrt(K.sum(K.square(x), axis=1)))

    def get_config(self):
        return {'C': float(self.l1)}

class Network(Model):

    def __init__(self, dim, batch_norm, dropout, rec_dropout, task,
                 target_repl=False, deep_supervision=False, num_classes=1,
                 depth=1, input_dim=76, multi=False, downstream_clients=1, 
                 output_dim=16, lambd=0.01, **kwargs):

        print("==> not used params in network class:", kwargs.keys())

        self.dim = dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.depth = depth
        self.lambd = lambd

        X = Input(shape=(None,input_dim), name='X') # input = concat of embeddings from downstream clients
        inputs = X
        y = X
        #for i in range(depth-1):
        #    initializer = tf.keras.initializers.RandomUniform(minval=-1., maxval=1., seed=583)
        #    if i == 0:
        #        y = Dense(64, kernel_regularizer=L21(0.01), kernel_initializer=initializer)(y)
        #    else:
        #        y = Dense(64, kernel_initializer=initializer)(y)
        #initializer = tf.keras.initializers.RandomUniform(minval=-1., maxval=1., seed=583)
        #y = Dense(output_dim, kernel_initializer=initializer)(y)
        for i in range(depth-1):
            if i == 0:
                y = Dense(self.dim, kernel_regularizer=L21(self.lambd))(y)
            else:
                y = Dense(self.dim)(y)
        if depth == 1:
            y = Dense(output_dim, kernel_regularizer=L21(self.lambd))(y)
        else:
            y = Dense(output_dim)(y)
        outputs = [y]

        super(Network, self).__init__(inputs=inputs, outputs=outputs)

    def say_name(self):
        return "{}.n{}{}{}{}.dep{}".format('k_lstm',
                                           self.dim,
                                           ".bn" if self.batch_norm else "",
                                           ".d{}".format(self.dropout) if self.dropout > 0 else "",
                                           ".rd{}".format(self.rec_dropout) if self.rec_dropout > 0 else "",
                                           self.depth)
    
#    def train_step(self, data):
#        x, y = data
#        with tf.GradientTape() as tape:
#            logits = self(x, training=True)
#            loss_value = self.compiled_loss(y, logits)
#        grads = tape.gradient(loss_value, self.trainable_variables)
#        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
#        self.compiled_metrics.update_state(y, logits)
#
#        return {m.name: m.result() for m in self.metrics}
