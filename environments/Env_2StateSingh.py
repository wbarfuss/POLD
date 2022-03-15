# -*- coding: utf-8 -*-
"""
2 state envrionment according to Singh et al., 1994
"""

import numpy as np

class TwoStateSingh(object):

    def __init__(self, obs_noise=0.0):
        self.N = 1  # starting with one agent, but this could be made adaptive
        self.M = 2  # 2 for now, but eventually three?
        self.Z = 2

        assert obs_noise >= 0.0
        self.noise = obs_noise
        if self.noise <= 0.5:
            self.Q = 2  # Agent can observe two state under noise
        else:
            self.Q = 1  # Agent perceives the two states as one

        self.T = self.TransitionTensor()
        self.R = self.RewardTensor()
        self.O = self.ObservationTensor()
        
        self.Aset = self.actions()
        self.Sset = self.states() 
        self.Oset = self.observations()
        
        self.state = 0 # inital state

    def states(self):
        return ['1', '2']
    
    def actions(self):
        return [['L', 'R']]
    
    def observations(self):
        if self.Q == 2:
            obs = [['o', 't']]  
        else:
            assert self.Q == 1
            obs = [['0']]
        return obs

    def obs_action_space(self):
        return np.zeros((self.Q, self.M))

    def TransitionTensor(self):
        """Get the Transition Tensor."""
        Tsas = np.zeros((2, 2, 2))

        Tsas[0,0,0] = 1
        Tsas[0,1,1] = 1
        Tsas[1,0,0] = 1
        Tsas[1,1,1] = 1

        return Tsas

    def RewardTensor(self):
        """Get the Reward Tensor R[i,s,a1,...,aN,s']."""
        Risas = np.zeros((1, 2, 2, 2))

        Risas[0,0,0,0] = -1
        Risas[0,0,1,1] = +1
        Risas[0,1,0,0] = +1
        Risas[0,1,1,1] = -1

        return Risas

    def ObservationTensor(self):

        if self.Q == 2:
            Oisao = np.zeros((1, 2, self.Q))

            Oisao[0,0,0] = 1 - self.noise
            Oisao[0,0,1] = 0 + self.noise
            Oisao[0,1,0] = 0 + self.noise
            Oisao[0,1,1] = 1 - self.noise
            
        else:
            assert self.Q == 1
            Oisao = np.zeros((1, 2, self.Q))
     
            Oisao[0,0,0] = 1.0
            Oisao[0,1,0] = 1.0
            
        return Oisao




    # Note: could be made availabe through general parent env  
    def step(self, jA):
        """
        iterate env for one step
        
        jA : joint action as an iterable
        """
        tps = self.T[tuple([self.state]+list(jA))].astype(float)
        # final state: if tps = 0 everywhere we arrived at a final state
        next_state = np.random.choice(range(len(tps)), p=tps)
        
        rewards = self.R[tuple([slice(self.N),self.state]+list(jA)
                               +[next_state])]
        self.state = next_state
        obs = self.observation()        
        
        return obs, rewards.astype(float), None, None

    # Note: could be made availabe through general parent env  
    def observation(self):

        OBS = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            ops = self.O[i, self.state]
            obs = np.random.choice(range(len(ops)), p=ops)
            OBS[i] = obs
        return OBS