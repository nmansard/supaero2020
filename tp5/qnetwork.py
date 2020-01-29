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


class QNetwork:
    '''
    Build a keras model computing:
    - qvalues(x) = [ Q(x,u_1) ... Q(x,u_NU) ]
    - value(x)   = max_u qvalues(x)
    - qvalue(x,u) = Q(x,u)
    '''
    def __init__(self,nx,nu,name='',nhiden=32):
        self.nx=nx;self.nu=nu
        input_x = keras.Input(shape=(nx,), name=name+'state')
        input_u = keras.Input(shape=(1,), name=name+'control',dtype="int32")
        dens1 = keras.layers.Dense(nhiden, activation='relu', name=name+'dense_1',
                                   bias_initializer='random_uniform')(input_x)
        dens2 = keras.layers.Dense(nhiden, activation='relu', name=name+'dense_2',
                                   bias_initializer='random_uniform')(dens1)
        qvalues = keras.layers.Dense(nu, activation='linear', name=name+'qvalues',
                                     bias_initializer='random_uniform')(dens2)
        value = keras.backend.max(qvalues,keepdims=True,axis=1)
        value = keras.layers.Lambda(lambda x:x,name=name+'value')(value)
        qvalue = batch_gather(qvalues,input_u)
        qvalue = keras.layers.Lambda(lambda x:x,name=name+'qvalue')(qvalue)
        policy = keras.backend.argmax(qvalues,axis=1)
        policy = keras.layers.Lambda(lambda x:x,name=name+'policy')(policy)
        
        #model = keras.Model(inputs=[input_x,input_u],outputs=[qvalues,value,qvalue,policy])
        self.trainer = keras.Model(inputs=[input_x,input_u],outputs=qvalue)
        self.trainer.compile(optimizer='adam',loss='mse')
        #self.trainer.optimizer.lr = LEARNING_RATE

        self.model = keras.Model(inputs=[input_x,input_u],
                                 outputs=[qvalues,value,qvalue,policy])

        self._policy = keras.backend.function(input_x,policy)
        self._qvalues = keras.backend.function(input_x,qvalues)
        self._value = keras.backend.function(input_x,value)

        # FOR DEBUG ONLY
        self._qvalues = keras.backend.function(input_x,qvalues)
        self._h1 = keras.backend.function(input_x,dens1)
        self._h2 = keras.backend.function(input_x,dens2)
        
    def targetAssign(self,target,rate):
        '''
        Change model to approach modelTarget, with homotopy parameter <rate>
        (rate=0: do not change, rate=1: exacttly set it to the target).
        '''
        assert(rate<=1 and rate>=0)
        for v,vtarget in zip(self.trainer.trainable_variables,target.trainer.trainable_variables):
            v.assign((1-rate)*v+rate*vtarget)
            #v.assign((1-rate)*v.eval()+rate*vtarget.eval())

    def policy(self,x,noise=None):
        if len(x.shape)==1: x=np.reshape(x,[1,len(x)])
        if noise is None:  return self._policy(x)
        q = self._qvalues(x)
        if noise is not None: q += (np.random.rand(self.nu)*2-1)*noise
        return np.argmax(q,axis=1)

    def value(self,x):
        if len(x.shape)==1: x=np.reshape(x,[1,len(x)])
        return self._value(x)
        

if __name__ == "__main__":
    NX = 3; NU = 10
    qnet = QNetwork(NX,NU)
    
    A = np.random.random([ NX,1])*2-1
    def data(x):
        y = (5*x+3)**2
        return y@A

    NSAMPLES = 1000
    xs = np.random.random([ NSAMPLES,NX ])
    us = np.random.randint( NU,size=NSAMPLES,dtype=np.int32 )
    ys = np.vstack([ data(x) for x in xs ])

    qnet.trainer.fit([xs,us],ys,epochs=50,batch_size=64)

    import matplotlib.pylab as plt
    plt.ion()
    plt.plot(xs,ys, '+')
    ypred=qnet.trainer.predict([xs,us])
    plt.plot(xs,ypred, '+r')
