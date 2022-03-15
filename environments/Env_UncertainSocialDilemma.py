"""
Two symmetric 2-agent 2-action matrix games

specified with the Prisoner's Dilemma nomenclature
R = reward of mutual cooperation
T = temptation of unilateral defection
S = sucker's payoff of unilateral cooperation
P = punishment of mutual defection
"""

import numpy as np
from .Env_Null import NullEnv

class UncertainSocialDilemma(NullEnv):

    def __init__(self, R1, T1, S1, P1, R2, T2, S2, P2, pC, obsnoise):
        self.N = 2
        self.M = 2
        self.Z = 2

        self.R1 = R1
        self.T1 = T1
        self.S1 = S1    
        self.P1 = P1    

        self.R2 = R2
        self.T2 = T2
        self.S2 = S2    
        self.P2 = P2    
        
        self.pC = pC  # prop. contract
        if not hasattr(obsnoise, "__iter__"):
            self.noise = np.array([obsnoise, obsnoise])
        else:
            assert len(obsnoise) == 2
            self.noise = np.array(obsnoise)
        assert min(self.noise) >= 0.0

        # --
        self.T = self.TransitionTensor()
        self.R = self.RewardTensor()
        self.O = self.ObservationTensor()
        self.state = 1 # inital state

    def actionticks(self):
        return [0, 1], ["coop.", "defect."]

    def stateticks(self):
        return [0, 1], ["no contract", "contract"]

    def TransitionTensor(self):
        """Get the Transition Tensor."""
        Tsas = np.ones((2, 2, 2, 2)) * (-1)

        Tsas[:, :, :, 0] = 1-self.pC
        Tsas[:, :, :, 1] = self.pC

        return Tsas

    def RewardTensor(self):
        """Get the Reward Tensor R[i,s,a1,...,aN,s']."""

        R = np.zeros((2, 2, 2, 2, 2))

        R[0, 0, :, :, 0] = [[self.R1, self.S1],
                            [self.T1, self.P1]]
        R[1, 0, :, :, 0] = [[self.R1, self.T1],
                            [self.S1, self.P1]]
        R[:, 0, :, :, 1] = R[:, 0, :, :, 0]

        R[0, 1, :, :, 1] = [[self.R2, self.S2],
                            [self.T2, self.P2]]
        R[1, 1, :, :, 1] = [[self.R2, self.T2],
                            [self.S2, self.P2]]
        R[:, 1, :, :, 0] = R[:, 1, :, :, 1]

        return R

    def ObservationTensor(self):

        if np.all(self.noise > 0.5):
            self.Q = 1
            Oiso = np.ones((self.N, self.Z, self.Q))
            
        else:
            self.Q = self.Z
            Oiso = np.zeros((self.N, self.Z, self.Q))

            for i in range(self.N):
                Oiso[i,0,0] = 1 - min(self.noise[i], 0.5)
                Oiso[i,0,1] = 0 + min(self.noise[i], 0.5)
                Oiso[i,1,0] = 0 + min(self.noise[i], 0.5)
                Oiso[i,1,1] = 1 - min(self.noise[i], 0.5)
            
        return Oiso