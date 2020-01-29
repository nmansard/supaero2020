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
import tensorflow.compat.v1 as tf1
import numpy as np

### --- Q-value networks

class QValueNetwork:
    '''
    We represent the Q function Q(x,u) --where u is an integer [0,NU-1]-- by
    N(x)=[ Q(x,u0) ... Q(x,u_NU-1) ].
    The policy is then Pi(x) = argmax N(x) and the value function is V(x) = max Q(x)

    The classical update rule with Q:
    max_theta Q(x,u;theta) - ( reward(x,u) + decay * max_u2 Q(x2,u2) )
    with x2 = f(x,u)
    
    then rewrite as:
    max_theta N(x;theta)[u] - (reward(x,u) + decay * max N(x2)
    '''
    
    def __init__(self,NX,NU,nhiden1=32,nhiden2=32,randomSeed=None):
        if randomSeed is None:
            import time
            randomSeed = int((time.time()%10)*1000)
        # n_init              = tflearn.initializations.truncated_normal(seed=randomSeed)
        #n_init              = tf.random.truncated_normal(seed=randomSeed)
        # u_init              = tflearn.initializations.uniform(minval=-0.003, maxval=0.003,\
        #                                                       seed=randomSeed)
        nvars           = len(tf.compat.v1.trainable_variables())

        #tflearn.input_data(shape=[None, NX])
        x       = tf.compat.v1.placeholder(tf.float32, [None, NX], name='state')

        #netx1   = tflearn.fully_connected(x,     nhiden1, weights_init=n_init, activation='relu')
        W1 = tf.Variable(tf.random.truncated_normal([NX, nhiden1], stddev=0.03), name='W1')
        b1 = tf.Variable(tf.random.truncated_normal([nhiden1]), name='b1')
        netx1 = tf.add(tf.matmul(x, W1), b1)
        netx1 = tf.nn.relu(netx1)

        #netx2   = tflearn.fully_connected(netx1, nhiden2, weights_init=n_init)
        W2 = tf.Variable(tf.random.truncated_normal([nhiden1, nhiden2], stddev=0.03), name='W2')
        b2 = tf.Variable(tf.random.truncated_normal([nhiden2]), name='b2')
        netx2 = tf.add(tf.matmul(netx1, W2), b2)
        netx2 = tf.nn.relu(netx2)
        
        #qvalues = tflearn.fully_connected(netx2, NU,      weights_init=u_init) # function of x only
        Wq = tf.Variable(tf.random.truncated_normal([nhiden2, NU], stddev=0.03), name='Wq')
        bq = tf.Variable(tf.random.truncated_normal([NU]), name='bq')
        qvalues = tf.add(tf.matmul(netx2, Wq), bq)
        #qvalues = tf.nn.relu(netxq)
       
        value   = tf.reduce_max(qvalues,axis=1)
        policy  = tf.argmax(qvalues,axis=1)

        #tflearn.input_data(shape=[None, 1], dtype=tf.int32)
        u       = tf.compat.v1.placeholder(tf.int32, [None, 1], name='control')

        bsize   = tf.shape(u)[0]
        idxs    = tf.reshape(tf.range(bsize),[bsize,1])
        ui      = tf.concat([idxs,u],1)
        print(ui)
        qvalue  = tf.gather_nd(qvalues,indices=ui)
        self.idxs = idxs
        self.ui = ui
        
        self.x          = x                                # Network state   <x> input in Q(x,u)
        self.u          = u                                # Network state   <x> input in Q(x,u)
        self.qvalue     = qvalue                           # Network output  <Q>
        self.value      = value                            # Optimal value function <Q*>
        self.policy     = policy                           # Greedy policy argmax<Q>
        self.qvalues    = qvalues                          # Q(x,.) = [ Q(x,0) ... Q(x,NU-1) ]
        self.variables  = tf.compat.v1.trainable_variables()[nvars:] # Variables to be trained
        self.hidens = [ netx1, netx2 ]                     # Hidden layers for debug

        # DG
        self.qvalues = qvalues
        self.h1 = netx1
        self.h2 = netx2
        
    def setupOptim(self,learningRate):
        qref            = tf1.placeholder(tf.float32, [None])
        #loss            = tflearn.mean_square(qref, self.qvalue)
        loss = tf.reduce_mean(tf.square(qref-self.qvalue))
        optim           = tf1.train.AdamOptimizer(learningRate).minimize(loss)

        self.qref       = qref          # Reference Q-values
        self.optim      = optim         # Optimizer
        return self

    def setupTargetAssign(self,nominalNet,updateRate):
        self.update_variables = \
            [ target.assign( updateRate*ref + (1-updateRate)*target )  \
                  for target,ref in zip(self.variables,nominalNet.variables) ]
        return self


