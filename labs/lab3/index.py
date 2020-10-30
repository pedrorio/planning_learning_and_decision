import numpy as np

def load_pomdp(file_path, g):
    file = np.load(file_path)
    X, A, Z, P, O, c = file['X'], file['A'], file['Z'], file['P'], file['O'], file['c']
    return X, A, Z, tuple(P), tuple(O), c, g



def gen_trajectory(POMDP, x0, n):
    X, A, Z, P, O, c, g = POMDP

    # P_A_{t+1} = P_A_{t} * P[a] * diag(O_A{o})

    # array of n+1 state indices
    # (29 + combinations of 2 keys + imp in 3 spaces)
    state_path = np.zeros(n+1, dtype=np.int16)
    state_path[0] = int(x0)
    x = x0
    action_path = np.zeros(n, dtype=np.int16)
    observation_path = np.zeros(n, dtype=np.int16)
    A_p = np.ones(len(A)) / len(A)
    A_p = np.expand_dims(A_p, axis=0)
    # print(A_p)
    # print(A_p.shape)
    belief = np.ones(len(X)) / len(X)
    belief = np.expand_dims(belief, axis=0)
    # print(X_p)
    # print(X_p.shape)

    
    
    for i in range(0,n):
        # array of n action indices
        # (5 actions, up, down, lef, right, listen)
        # print(f'i is {i}')
        # print("choosing a")
        
        a = np.random.choice(len(A), p=A_p[0, :])
        action_path[i] = a
        # print("adding a to action path")
        # print(f'a is: {a}')

        Z_p = O[a][x, :]
        z = int(np.random.choice(len(Z), p=Z_p))
        observation_path[i] = z
        # print("adding z to observation path")
        # print(f'z is: {z}')

        # print('computing belief')
        # print(f'belief: {belief.shape}')
        # print(f'P[a]: { P[a].shape}')
        # print(f'np.diag(O[a][:,z]): {np.diag(O[a][:, z]).shape}')
        # print(f'O[a]: {O[a].shape}')
        
        # X_p = np.matmul(np.matmul(X_p,P[a]),np.diag(O[a][x,:]))
        belief_est = np.matmul(belief, P[a])
        belief = np.matmul(belief_est, np.diag(O[a][:, z]))

        # belief = np.matmul(belief_est, belief)

        # print(np.sum(np.abs(X_p), axis=1))
        # X_p /= np.sum(np.abs(X_p), axis=1)
        # print(belief)

        norm = np.linalg.norm(belief, axis=1, keepdims=True, ord=1)
        # print(norm)
        belief = belief/norm
        # print(belief)
        
        # print(belief.shape)
        x = np.random.choice(len(X), p=belief[0, :])
        
        state_path[i+1] = x
        # print("adding x to state path")
        # print(f'x is: {x}')
        # print("\n")

        # [state_path[0]]
        # print(TP_A)

        # array of n observation indices
        
        # O_A = O[A_i]
        # print(O_A.shape)
    return state_path, action_path, observation_path
    


M = load_pomdp('maze.npz', 0.95)
t = gen_trajectory(M, 0,  10)
    
