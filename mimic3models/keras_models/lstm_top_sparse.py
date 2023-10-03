from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Masking, Dropout
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
                 depth=1, input_dim=76, multi=False, downstream_clients=1, output_dim=64, lambd=0.01, **kwargs):

        print("==> not used params in network class:", kwargs.keys())

        self.dim = dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.depth = depth
        self.lambd = lambd

        if num_classes == 1:
            final_activation = 'sigmoid'
        else:
            final_activation = 'softmax'

        #L = Input(shape=dim*downstream_clients, name='X') # input = concat of embeddings from downstream clients
        L = Input(shape=output_dim, name='X') # input = concat of embeddings from downstream clients

        # Output module of the network
        if dropout > 0:
            L = Dropout(dropout)(L)

        if target_repl:
            y = TimeDistributed(Dense(num_classes, activation=final_activation),
                                name='seq')(L)
            y_last = LastTimestep(name='single')(y)
            outputs = [y_last, y]
        elif deep_supervision:
            y = TimeDistributed(Dense(num_classes, activation=final_activation))(L)
            y = ExtendMask()([y, M])  # this way we extend mask of y to M
            outputs = [y]
        else:
            y = None
            if multi:
                y = Dense(num_classes, kernel_regularizer=L21(self.lambd))(L)
            else:
                y = Dense(num_classes, activation=final_activation, kernel_regularizer=L21(self.lambd))(L)
            outputs = [y]

        super(Network, self).__init__(inputs=L, outputs=outputs)

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
