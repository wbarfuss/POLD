"""
Parent class for environments
"""
import numpy as np

class NullEnv(object):
    
    def __init__(self):
        self.N = 0
        self.M = 0  
        self.Z = 0
        self.Q = 0
     
        self.state = 0
                
        self.T = self.TransitionTensor()
        self.R = self.RewardTensor()
        self.O = self.ObservationTensor()
        
        
    def TransitionTensor(self):
        raise NotImplementedError
    
    def RewardTensor(self):
        raise NotImplementedError
    
    def ObservationTensor(self):
        raise NotImplementedError
    
    def obs_action_space(self):
        return np.zeros((self.Q, self.M))
    
    def step(self, jA):
        """
        Iterate the environment for one step
        
        jA : joint action as an iterable of agents
        """
        tps = self.T[tuple([self.state]+list(jA))].astype(float)
        # final state: if tps = 0 everywhere we arrived at a final state
        next_state = np.random.choice(range(len(tps)), p=tps)
        
        rewards = self.R[tuple([slice(self.N),self.state]+list(jA)
                               +[next_state])]
        self.state = next_state
        obs = self.observation()        
        
        return obs, rewards.astype(float)

    # Note: could be made availabe through general parent env  
    def observation(self):

        OBS = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            ops = self.O[i, self.state]
            obs = np.random.choice(range(len(ops)), p=ops)
            OBS[i] = obs
        return OBS