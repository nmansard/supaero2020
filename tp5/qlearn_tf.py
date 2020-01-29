'''
Train a Q-value following a classical Q-learning algorithm (enforcing the
satisfaction of HJB method), using a noisy greedy exploration strategy.

The result of a training for a continuous Cozmo are stored in netvalue/qlearn_cozmo1.ckpt.

Reference:
Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." 
Nature 518.7540 (2015): 529.
'''

#from cozmomodel import Cozmo1 as Env
from env_pendulum import EnvPendulumHybrid as Env
from qnetwork_tf import QValueNetwork
from collections import deque
import time
import signal
import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import numpy as np

tf.compat.v1.disable_eager_execution()

### --- Random seed
RANDOM_SEED = int((time.time()%10)*1000)
print("Seed = %d" %  RANDOM_SEED)
np .random.seed     (RANDOM_SEED)
#tf .set_random_seed (RANDOM_SEED)
random.seed         (RANDOM_SEED)

### --- Hyper paramaters
NEPISODES               = 10000           # Max training steps
NSTEPS                  = 60           # Max episode length
QVALUE_LEARNING_RATE    = 0.001         # Base learning rate for the Q-value Network
DECAY_RATE              = 0.99          # Discount factor 
UPDATE_RATE             = 0.01          # Homotopy rate to update the networks
REPLAY_SIZE             = 10000         # Size of replay buffer
BATCH_SIZE              = 64            # Number of points to be fed in stochastic gradient
NH1 = NH2               = 32            # Hidden layer size

### --- Environment
env                 = Env()
NX                  = env.nx            # ... training converges with q,qdot with 2x more neurones.
NU                  = env.nu            # Control is dim-1: joint torque

### --- Replay memory
class ReplayItem:
    def __init__(self,x,u,r,d,x2):
        self.x          = x
        self.u          = u
        self.reward     = r
        self.done       = d
        self.x2         = x2

replayDeque = deque()

### --- Tensor flow initialization
qvalue          = QValueNetwork(NX=NX,NU=NU,nhiden1=NH1,nhiden2=NH2,randomSeed=RANDOM_SEED)
qvalueTarget    = QValueNetwork(NX=NX,NU=NU,nhiden1=NH1,nhiden2=NH2,randomSeed=RANDOM_SEED)
qvalue      . setupOptim       (learningRate=QVALUE_LEARNING_RATE)
qvalueTarget. setupTargetAssign(qvalue,updateRate=UPDATE_RATE)

sess            = tf1.InteractiveSession()
tf1.global_variables_initializer().run()

# Uncomment to restore networks
#tf1.train.Saver().restore(sess, "netvalues/qlearn_tf1.ckpt"); NEPISODES =100

def noisygreedy(x,rand=None):
    q = sess.run(qvalue.qvalues,feed_dict={ qvalue.x: np.matrix(x) })
    if rand is not None: q += (np.random.rand(env.nu)*2-1)*rand
    return np.argmax(q)

def rendertrial(maxiter=NSTEPS,verbose=True):
    x = env.reset()
    rsum = 0.
    for i in range(maxiter):
        u = sess.run(qvalue.policy,feed_dict={ qvalue.x: np.matrix(x) })
        x, reward = env.step(u)
        env.render()
        time.sleep(1e-2)
        rsum += reward
    if verbose: print('Lasted ',i,' timestep -- total reward:',rsum)
signal.signal(signal.SIGTSTP, lambda x,y:rendertrial()) # Roll-out when CTRL-Z is pressed

### History of search
h_rwd = []
h_ste = []    

### --- Training
for episode in range(1,NEPISODES):
    x    = env.reset()
    rsum = 0.0

    for step in range(NSTEPS):
        u       = noisygreedy(x,                                    # Greedy policy ...
                              rand=1. / (1. + episode + step))      # ... with noise
        x2,r    = env.step(u)
        done    = r>0

        replayDeque.append(ReplayItem(x,u,r,done,x2))                # Feed replay memory ...
        if len(replayDeque)>REPLAY_SIZE: replayDeque.popleft()       # ... with FIFO forgetting.

        assert( x2.shape == (NX,) )
        
        rsum   += r
        x       = x2
        if done: break
        
        # Start optimizing networks when memory size > batch size.
        if len(replayDeque) > BATCH_SIZE:     
            batch = random.sample(replayDeque,BATCH_SIZE)            # Random batch from replay memory.
            x_batch    = np.vstack([ b.x      for b in batch ])
            u_batch    = np.vstack([ b.u      for b in batch ])
            r_batch    = np. stack([ b.reward for b in batch ])
            d_batch    = np. stack([ b.done   for b in batch ])
            x2_batch   = np.vstack([ b.x2     for b in batch ])

            # Compute Q(x,u) from target network
            v_batch    = sess.run(qvalueTarget.value, feed_dict={ qvalueTarget.x : x2_batch })
            qref_batch = r_batch + (d_batch==False)*(DECAY_RATE*v_batch)

            # Update qvalue to solve HJB constraint: q = r + q'
            sess.run(qvalue.optim, feed_dict={ qvalue.x    : x_batch,
                                               qvalue.u    : u_batch,
                                               qvalue.qref : qref_batch })

            # Update target networks by homotopy.
            sess.run(qvalueTarget.update_variables)

    # \\\END_FOR step in range(NSTEPS)

    # Display and logging (not mandatory).
    print('Ep#{:3d}: lasted {:d} steps, reward={:3.0f}' .format(episode, step,rsum))
    h_rwd.append(rsum)
    h_ste.append(step)
    if not (episode+1) % 200:     rendertrial(30)

# \\\END_FOR episode in range(NEPISODES)

print("Average reward during trials: %.3f" % (sum(h_rwd)/NEPISODES))
rendertrial()
plt.plot( np.cumsum(h_rwd)/range(1,NEPISODES) )
plt.show()

# Uncomment to save networks
#tf.train.Saver().save   (sess, "netvalues/qlearn_cozmo1.ckpt")
