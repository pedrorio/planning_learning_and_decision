#!/usr/bin/env python
import pdb



import numpy as np

mdp_info = np.load('doom.npz', allow_pickle=True)

# The MDP is a tuple (X, A, P, c, gamma)
M = mdp_info['M']

# We also load the optimal Q-function for the MDP
Qopt = mdp_info['Q']

print(M)


import numpy as np
import numpy.random as rnd


def sample_transition(MDP, s, a):
    c = MDP[3][s, a]
    prob = MDP[2][a][s]
    snew = rnd.choice(np.arange(0, len(MDP[0])), p=prob)
    
    return (s, a, c, snew)
    



rnd.seed(42)

# Select random state and action
s = rnd.randint(len(M[0]))
a = rnd.randint(len(M[1]))

s, a, cnew, snew = sample_transition(M, s, a)

print('Observed transition:\n(', end='')
print(M[0][s], end=', ')
print(M[1][a], end=', ')
print(cnew, end=', ')
print(M[0][snew], end=')\n')

# Select random state and action
s = rnd.randint(len(M[0]))
a = rnd.randint(len(M[1]))

s, a, cnew, snew = sample_transition(M, s, a)

print('Observed transition:\n(', end='')
print(M[0][s], end=', ')
print(M[1][a], end=', ')
print(cnew, end=', ')
print(M[0][snew], end=')\n')

# Select random state and action
s = rnd.randint(len(M[0]))
a = rnd.randint(len(M[1]))

s, a, cnew, snew = sample_transition(M, s, a)

print('Observed transition:\n(', end='')
print(M[0][s], end=', ')
print(M[1][a], end=', ')
print(cnew, end=', ')
print(M[0][snew], end=')\n')




# Add your code here.
def egreedy(Q, eps=0.1):
    N = Q.shape[0]
    realisation = rnd.rand()
    
    # pdb.set_trace()
    
    if realisation<=eps:
        return rnd.randint(N)
    else:
        return rnd.choice(np.argmin(Q).flatten(),1)[0]
        # minimum = np.array(np.argmin(Q)).flatten()
        # return rnd.choice(minimum)


s = 51
a = egreedy(Qopt[s, :], eps=0.0)
print('State:', M[0][s], '- action:', M[1][a])
a = egreedy(Qopt[s, :], eps=0.5)
print('State:', M[0][s], '- action:', M[1][a])
a = egreedy(Qopt[s, :], eps=1.0)
print('State:', M[0][s], '- action:', M[1][a])

s = 71
a = egreedy(Qopt[s, :], eps=0.0)
print('State:', M[0][s], '- action:', M[1][a])
a = egreedy(Qopt[s, :], eps=0.5)
print('State:', M[0][s], '- action:', M[1][a])
a = egreedy(Qopt[s, :], eps=1.0)
print('State:', M[0][s], '- action:', M[1][a])

s = 23
a = egreedy(Qopt[s, :], eps=0.0)
print('State:', M[0][s], '- action:', M[1][a])
a = egreedy(Qopt[s, :], eps=0.5)
print('State:', M[0][s], '- action:', M[1][a])
a = egreedy(Qopt[s, :], eps=1.0)
print('State:', M[0][s], '- action:', M[1][a])


# Add your code here.

def mb_learning(M, n, qinit, Pinit, cinit):
    n_states = len(M[0])
    n_actions = len(M[1])
    Q = qinit
    P = Pinit
    c = cinit
    s = rnd.randint(n_states)
    Nt = np.zeros((n_states, n_actions))
    step = 0
    
    while step < n:
        a = egreedy(Q[s, :], 0.15) #Choosing the action with egreedy
        Nt[s,a] += 1
        
        s, a , cnew, snew = sample_transition(M, s, a) #sample the transition
        
        #Update from (s,a)
        alphat = (1/(Nt[s,a]+1))
        P[a][s] = P[a][s] - alphat * P[a][s]
        P[a][s][snew] = P[a][s][snew] + alphat
        c[s,a] = c[s,a] + alphat * (cnew - c[s,a])
        Q[s][a] = c[s,a] + M[4] * np.sum(np.multiply(P[a][s],np.min(Q, axis=1)))
        
        step = step + 1
        s = snew
    return(Q, P, c)



rnd.seed(42)

# Initialize transition probabilities
pinit = ()

for a in range(len(M[1])):
    pinit += (np.eye(len(M[0])),)

# Initialize cost function
cinit = np.zeros((len(M[0]), len(M[1])))

# Initialize Q-function
qinit = np.zeros((len(M[0]), len(M[1])))

# Run 1000 steps of model-based learning
qnew, pnew, cnew = mb_learning(M, 1000, qinit, pinit, cinit)

# Compare the learned Q with the optimal Q
print('Error in Q after 1000 steps:', np.linalg.norm(qnew - Qopt))

# Run 1000 additional steps of model-based learning
qnew, pnew, cnew = mb_learning(M, 1000, qnew, pnew, cnew)

# Compare once again the learned Q with the optimal Q
print('Error in Q after 2000 steps:', np.linalg.norm(qnew - Qopt))

# Add your code here.
def qlearning(M, n, qinit):
    n_states = len(M[0])
    Q = qinit
    
    s = rnd.randint(n_states)
    alpha = 0.3
    step = 0
    
    while step < n:
        a = egreedy(Q[s, :], 0.15) #Choosing the action with egreedy
        s, a , cnew, snew = sample_transition(M, s, a) #sample the transition
        
        #Update Q-function
        Q[s][a] = Q[s][a] + alpha * (cnew + M[4] * np.min(Q[snew, :]) - Q[s][a])
        
        step = step + 1
        s = snew
        
    return Q

rnd.seed(42)

# Initialize Q-function
qinit = np.zeros((len(M[0]), len(M[1])))

# Run 1000 steps of model-based learning
qnew = qlearning(M, 1000, qinit)

# Compare the learned Q with the optimal Q
print('Error in Q after 1000 steps:', np.linalg.norm(qnew - Qopt))

# Run 1000 additional steps of model-based learning
qnew = qlearning(M, 1000, qnew)

# Compare once again the learned Q with the optimal Q
print('Error in Q after 2000 steps:', np.linalg.norm(qnew - Qopt))

# Add your code here.
def sarsa(M, n, qinit):
    n_states = len(M[0])
    Q = qinit
    
    s = rnd.randint(n_states)
    a = egreedy(Q[s, :], 0.15)
    alpha = 0.3
    step = 0
    
    while step < n:
        s, a , cnew, snew = sample_transition(M, s, a) #sample the transition
        
        anew = egreedy(Q[snew, :], 0.15) #New action
        
        Q[s][a] = Q[s][a] + alpha * (cnew + M[4] * Q[snew][anew] - Q[s][a])
        
        step = step + 1
        a = anew
        s = snew
        
    return Q


rnd.seed(42)

# Initialize Q-function
qinit = np.zeros((len(M[0]), len(M[1])))

# Run 1000 steps of model-based learning
qnew = sarsa(M, 1000, qinit)

# Compare the learned Q with the optimal Q
print('Error in Q after 1000 steps:', np.linalg.norm(qnew - Qopt))

# Run 1000 additional steps of model-based learning
qnew = sarsa(M, 1000, qnew)

# Compare once again the learned Q with the optimal Q
print('Error in Q after 2000 steps:', np.linalg.norm(qnew - Qopt))


# get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

STEPS = 10
ITERS = 10000
RUNS  = 10

iters = range(0, STEPS * ITERS + 1, STEPS)

# Error matrices
Emb = np.zeros(ITERS + 1)
Eql = np.zeros(ITERS + 1)
Ess = np.zeros(ITERS + 1)

Emb[0] = np.linalg.norm(Qopt) * RUNS
Eql[0] = Emb[0]
Ess[0] = Emb[0]

for n in range (RUNS):

    # Initialization
    pmb = ()
    for a in range(len(M[1])):
        pmb += (np.eye(len(M[0])),)
    cmb = np.zeros((len(M[0]), len(M[1])))
    qmb = np.zeros((len(M[0]), len(M[1])))

    qql = np.zeros((len(M[0]), len(M[1])))

    qss = np.zeros((len(M[0]), len(M[1])))

    # Run evaluation
    for t in range(ITERS):
        qmb, pmb, cmb = mb_learning(M, STEPS, qmb, pmb, cmb)
        Emb[t + 1] += np.linalg.norm(Qopt - qmb)

        qql = qlearning(M, STEPS, qql)
        Eql[t + 1] += np.linalg.norm(Qopt - qql)

        qss = sarsa(M, STEPS, qss)
        Ess[t + 1] += np.linalg.norm(Qopt - qss)


plt.figure()
plt.plot(iters, Emb, label='Model based learning')
plt.plot(iters, Eql, label='Q-learning')
plt.plot(iters, Ess, label='SARSA')
plt.legend()
plt.xlabel('N. iterations')
plt.ylabel('Error in $Q$-function')
plt.show()