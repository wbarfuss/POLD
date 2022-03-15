# -*- coding: utf-8 -*-
import numpy as np
from agents.deterministic import detMAE
from environments.Env_ReRe import ReRe as ENV


def value_iteration(R, T, gamma):
    """
    Find optimal policy of a markov decision process.
    
    Taken from Sutton and Barto, Figure 4.5

    Parameters
    ----------
    R : np.array
        Reward tensor.
    T : np.array
        Transition tensor.

    Returns
    -------
    X : np.array
        Optimal deterministic policy
    """
    assert R.shape[0] == 1, "No games"
    Z = R.shape[1]  # Nr of states
    M = R.shape[2]  # Nr of actions
    
    V = np.zeros(Z)
    
    delt = 1; eps = 0.000001
    n = np.newaxis
    while delt > eps:
        delt = 0
        for s in range(Z):
            v = V[s]
            V[s] = np.max(np.sum(T[s, :, :] * (R[0,s,:,:] + gamma*V[n, :]),
                                 axis=-1), axis=-1)
            delt = max(delt, np.abs(v - V[s]))
    
    # print(V)
    pi = np.zeros(Z, dtype=int)
    X = np.zeros((1, Z, M))

    for s in range(Z):
        pi[s] = np.argmax(np.sum(T[s, :, :] * (R[0,s,:,:] + gamma*V[n, :]),
                                 axis=-1), axis=-1)
        
        X[0, s, pi[s]] = 1
        
    return X

def avgreward_value_iteration(R, T):
    Z = R.shape[1]  # Nr of states
    M = R.shape[2]  # Nr of actions
    avgR = 0
    V = np.zeros(Z)
    
    delt = 1; eps = 0.000001
    n = np.newaxis
    while delt > eps:
        delt = 0
        for s in range(Z):
            v = V[s]
            V[s] = np.max(np.sum(T[s, :, :] * (R[0,s,:,:] - avgR
                                               + V[n, :]),
                                 axis=-1), axis=-1)
            delt = max(delt, np.abs(v - V[s]))
        pi = _from_V_to_pi(V, T, R, avgR)
        X = _from_pi_to_X(pi, M)
        avgR = average_reward(X, R, T)
    return X
    
            
def _from_V_to_pi(V, T, R, avgR):
    Z = len(V)
    pi = np.zeros(Z, dtype=int)
    n = np.newaxis
    for s in range(Z):
        pi[s] = np.argmax(np.sum(T[s, :, :] * (R[0,s,:,:] - avgR + V[n, :]),
                                 axis=-1), axis=-1)
    return pi

def _from_pi_to_X(pi, M):
    Z = len(pi)
    X = np.zeros((1, Z, M))

    for s in range(Z):
        X[0, s, pi[s]] = 1
        
    return X  


def average_reward(X, R, T, gamma=0.9):
    alpha = 0.02
    beta = 250
    
    Z = X.shape[1]  # Nr of states
    M = X.shape[2]  # Nr of actions

    O = np.zeros((1, Z, Z))
    O[0, :, :] = np.eye(Z)
    agents = detMAE(T, R, O, alpha, beta, gamma)
    
    Rio = agents.obtain_Rio(X)
    pio = agents.obtain_obsdist(X)
    return np.einsum(pio, [0,1], Rio, [0, 1], [0])[0]



def get_optimal_average_rewards(Cs=[3,4,5,6,7,8,9], growthrate=0.8,
                                dE=0.2, sig=0.25):
    
    Rs = []
    for capacity in Cs:
        env = ENV(r=growthrate, C=capacity, obs=None, deltaE=dE, sig=sig)
        T = env.TransitionTensor().astype(float); # print(T.sum(-1))
        R = env.RewardTensor().astype(float)
        O = env.ObservationTensor(); # print(O.sum(-1))
        
        AvgOptX = avgreward_value_iteration(R, T)
        avgR = average_reward(AvgOptX, R, T)
        
        Rs.append(avgR)
    return Rs