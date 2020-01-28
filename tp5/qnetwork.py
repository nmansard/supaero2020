'''
Deep Q learning, i.e. learning the Q function Q(x,u) so that Pi(x) = u = argmax Q(x,u)
is the optimal policy. The control u is discretized as 0..NU-1

This program instantiates an environment env and a Q network qvalue.
The main signals are qvalue.x (state input), qvalue.qvalues (value for any u in 0..NU-1),
qvalue.policy (i.e. argmax(qvalue.qvalues)) and qvalue.qvalue (i.e. max(qvalue.qvalue)).

Reference:
Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." 
Nature 518.7540 (2015): 529.
'''

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import tensorflow.keras as keras
import tensorflow.keras.backend as K

def batch_gather(reference, indices):
    """
    From https://github.com/keras-team/keras/pull/6377 (not merged).

    Batchwise gathering of row indices.
    The numpy equivalent is `reference[np.arange(batch_size), indices]`, where
    `batch_size` is the first dimension of the reference tensor.
    # Arguments
        reference: A tensor with ndim >= 2 of shape.
          (batch_size, dim1, dim2, ..., dimN)
        indices: A 1d integer tensor of shape (batch_size) satisfying
          0 <= i < dim2 for each element i.
    # Returns
        The selected tensor with shape (batch_size, dim2, ..., dimN).
    # Examples
        1. If reference is `[[3, 5, 7], [11, 13, 17]]` and indices is `[2, 1]`
        then the result is `[7, 13]`.
    """
    batch_size   = keras.backend.shape(reference)[0]
    indices      = tf.concat([tf.reshape(tf.range(batch_size),[batch_size,1]),
                              indices],1)
    return tf.gather_nd(reference,indices=indices)


NX=3
NU=11
NHIDEN1=32
NHIDEN2=32

x = np.random.random([1,NX])
xs = np.random.random([5,NX])
u = np.array([1])
us = np.array([1,1,1,2,0])
usi = np.array([range(5),us]).T

model = keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(keras.layers.Dense(NX, activation='relu'))
model.add(keras.layers.Dense(NHIDEN1, activation='relu'))
model.add(keras.layers.Dense(NHIDEN2, activation='relu'))
model.add(keras.layers.Dense(NU, activation='linear'))

input_x = keras.Input(shape=(NX,), name='state')
dens1 = keras.layers.Dense(NHIDEN1, activation='relu', name='dense_1')(input_x)
dens2 = keras.layers.Dense(NHIDEN2, activation='relu', name='dense_2')(dens1)
qvalues = keras.layers.Dense(NU, activation='linear', name='values')(dens2)
#value = tf.reduce_max(qvalues)
value = keras.backend.max(qvalues,keepdims=True,axis=1)

input_u = keras.Input(shape=(1,), name='control',dtype="int32")
bsize   = K.shape(input_u)[0]
idxs    = tf.reshape(tf.range(bsize),[bsize,1])
ui      = tf.concat([idxs,input_u],1)
qvalue  = tf.gather_nd(qvalues,indices=ui)


qvalue = batch_gather(qvalues,input_u)
#qvalue = tf.gather_nd(input_x,tf.stack([tf.range(K.shape(input_u)[0]), input_u], axis=1))
# qvalue = tf.gather_nd(qvalues,
#                       indices=tf.concat([ tf.reshape(tf.range(tf.shape(input_u)[0],input_u],1)


model2 = keras.Model(inputs=input_x, outputs=qvalues)
modelmax = keras.Model(inputs=input_x, outputs=value)
modelqv = keras.Model(inputs=input_x, outputs=[qvalues,value])
modelq = keras.Model(inputs=[input_x,input_u],outputs=[qvalue])

qs,v=modelqv.predict(x)
assert(max(qs.flat)==v[0,0])

qs,v=modelqv.predict(xs)
q=modelq.predict([xs,us])
for i,ui in enumerate(us):
    assert(qs[i,ui]==q[i])


stophere

n_init              = tflearn.initializations.truncated_normal(seed=randomSeed)
u_init              = tflearn.initializations.uniform(minval=-0.003, maxval=0.003,\
                                                              seed=randomSeed)
nvars           = len(tf.trainable_variables())

#x       = tflearn.input_data(shape=[None, NX])
#netx1   = tflearn.fully_connected(x,     nhiden1, weights_init=n_init, activation='relu')
#netx2   = tflearn.fully_connected(netx1, nhiden2, weights_init=n_init)
qvalues = tflearn.fully_connected(netx2, NU,      weights_init=u_init) # function of x only
value   = tf.reduce_max(qvalues,axis=1)
policy  = tf.argmax(qvalues,axis=1)

u       = tflearn.input_data(shape=[None, 1], dtype=tf.int32)
bsize   = tf.shape(u)[0]
idxs    = tf.reshape(tf.range(bsize),[bsize,1])
ui      = tf.concat([idxs,u],1)
qvalue  = tf.gather_nd(qvalues,indices=ui)
