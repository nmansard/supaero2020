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


def buildQModel(nx,nu,nhiden=32):
    '''
    Build a keras model computing:
    - qvalues(x) = [ Q(x,u_1) ... Q(x,u_NU) ]
    - value(x)   = max_u qvalues(x)
    - qvalue(x,u) = Q(x,u)
    '''
    input_x = keras.Input(shape=(nx,), name='state')
    input_u = keras.Input(shape=(1,), name='control',dtype="int32")
    dens1 = keras.layers.Dense(nhiden, activation='relu', name='dense_1',
                               bias_initializer='random_uniform')(input_x)
    dens2 = keras.layers.Dense(nhiden, activation='relu', name='dense_2',
                               bias_initializer='random_uniform')(dens1)
    qvalues = keras.layers.Dense(nu, activation='linear', name='values',
                                 bias_initializer='random_uniform')(dens2)
    value = keras.backend.max(qvalues,keepdims=True,axis=1)
    qvalue = batch_gather(qvalues,input_u)
    policy = keras.backend.argmax(qvalues,axis=1)
    
    model_qs = keras.Model(inputs=input_x,outputs=qvalues)
    model_v = keras.Model(inputs=input_x,outputs=value)
    model_q = keras.Model(inputs=[input_x,input_u],outputs=qvalue)
    model_pi = keras.Model(inputs=input_x,outputs=policy)
    model_qs.compile(optimizer='adam',loss='mse')
    model = keras.Model(inputs=[input_x,input_u],outputs=[qvalues,value,qvalue,policy])
    
    return model_qs,model_v,model_q,model_pi

def targetAssign(model,modelTarget,rate):
    '''
    Change model to approach modelTarget, with homotopy parameter <rate>
    (rate=0: do not change, rate=1: exacttly set it to the target).
    '''
    assert(rate<=1 and rate>=0)
    for v,vtarget in zip(model.trainable_variables,modelTarget.trainable_variables):
        v.assign((1-rate)*v.value()+rate*vtarget.value())
    

if __name__ == "__main__":
    NX = NU = 1
    model,_,_,policy = buildQModel(NX,NU)
    model2,_,_,_ = buildQModel(NX,NU)
    
    A = np.random.random([ NX,NU])*2-1
    def data(x):
        y = (5*x+3)**2
        return y@A

    NSAMPLES = 1000
    xs = np.random.random([ NSAMPLES,NX ])
    ys = np.vstack([ data(x) for x in xs ])

    model.fit(xs,ys,epochs=200,batch_size=64)

    import matplotlib.pylab as plt
    plt.ion()
    plt.plot(xs,ys, '+')
    ypred=model.predict(xs)
    plt.plot(xs,ypred, '+r')
