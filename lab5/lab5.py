import numpy as np
import numpy as np
import numpy.random as rnd

mdp_info = np.load('doom.npz', allow_pickle=True)

# The MDP is a tuple (X, A, P, c, gamma)
M = mdp_info['M']

# We also load the optimal Q-function for the MDP
Qopt = mdp_info['Q']

print(M)

def sample_transition(MDP, s, a):
    c = MDP[3][s, a]
    prob = MDP[2][a][s]
    snew = rnd.choice(np.arange(0, len(MDP[0])), p=prob)

    return (s, a, c, snew)

def egreedy(Q, eps=0.1):
    N = Q.shape[0]
    u = rnd.rand()

    if u<eps:
        return rnd.randint(N)
    else:
        minimum = np.array(np.argmin(Q)).flatten()
        return rnd.choice(minimum)


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

# Comments:

# Looking at the graph we can see that the model based learning is the one who gets not only the lowest possible error but also the fastest one to get there, this can be explained because for each time step the model improves its estimates of the probability matrix P, the cost matrix c and the Q-function.
# The Q-learning algorithm is an optimistic model that tells an agent what action to take under what circumstances.
# In the SARSA the agent interacts with the environment and updates the policy based on actions taken, therefore this is known as an on-policy learning algorithm.
# While SARSA learns the Q values associated with taking the policy it follows itself, Q-learning learns the Q values associated with taking the optimal policy while following an exploration/exploitation policy.
