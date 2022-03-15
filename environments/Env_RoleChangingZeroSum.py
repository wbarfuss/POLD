"""
The 2-state Matching Pennies according to HennesEtAl2010
"""
# import sys
# from pathlib import Path
# base_dir = Path(__file__).resolve().parent.parent.parent
# sys.path.append(str(base_dir))
# from LearningDynamics.Envs.Env_Null import NullEnv
from .Env_Null import NullEnv
import numpy as np

class RoleChangingZeroSum(NullEnv):

    def __init__(self, obsnoise):
        
        if not hasattr(obsnoise, "__iter__"):
            self.noise = np.array([obsnoise, obsnoise])
        else:
            assert len(obsnoise) == 2
            self.noise = np.array(obsnoise)
        assert min(self.noise) >= 0.0
        
        self.N = 2
        self.M = len(self.actions())
        self.Z = len(self.states())
        self.Q = len(self.observations())
        
        # -- 
        self.T = self.TransitionTensor()
        self.R = self.RewardTensor()
        self.state = 1 # inital state

    def actions(self):
        acts = ['a', 'b']
        return acts
    
    def states(self):
        states = ['X', 'Y']
        return states
    
    def observations(self):
        if not np.all(self.noise > 0.5):
             obs = ['x', 'y']
        else:
            obs = ['z']
        return obs    
        
    def FinalStates(self):
        return [0, 0]

    def TransitionTensor(self):
        """Get the Transition Tensor."""
        Tsas = np.ones((2, 2, 2, 2)) * (-1)

        #investiagte
        # T1 = np.array([[1.0, 1.0],
        #                [0.0, 0.0]])
        # T2 = np.array([[0.0, 0.0],
        #                [1.0, 1.0]])
        
        # T1 = np.array([[0.0, 1.0],  # from state 0 to state 1
        #                [1.0, 0.0]])
        # T2 = np.array([[1.0, 0.0],  # from state 1 to state 0
        #                [0.0, 1.0]])
        
        T1 = np.array([[1.0, 1.0],  # from state 0 to state 1
                       [0.0, 0.0]])
        T2 = np.array([[0.0, 0.0],  # from state 1 to state 0
                       [1.0, 1.0]])
        
        Tsas[0, :, :, 1] = T1
        Tsas[0, :, :, 0] = 1-T1
        Tsas[1, :, :, 0] = T2
        Tsas[1, :, :, 1] = 1-T2
        
        return Tsas

    def RewardTensor(self):
        """Get the Reward Tensor R[i,s,a1,...,aN,s']."""

        R = np.zeros((2, 2, 2, 2, 2))

        R[0, 0, :, :, 0] = [[1 , 0 ],
                            [0 , 1 ]]
        R[1, 0, :, :, 0] = [[0 , 1 ],
                            [1 , 0 ]]

        R[:, 0, :, :, 1] = R[:, 0, :, :, 0]

        R[0, 1, :, :, 1] = [[0 , 1 ],
                            [1 , 0 ]]
        R[1, 1, :, :, 1] = [[1 , 0 ],
                            [0 , 1 ]]

        R[:, 1, :, :, 0] = R[:, 1, :, :, 1]

        return R


    def ObservationTensor(self):

        if np.all(self.noise > 0.5):
            #self.Q = 1
            Oiso = np.ones((self.N, self.Z, self.Q))
    
        else:
            #self.Q = self.Z
            Oiso = np.zeros((self.N, self.Z, self.Q))

            for i in range(self.N):
                Oiso[i,0,0] = 1 - min(self.noise[i], 0.5)
                Oiso[i,0,1] = 0 + min(self.noise[i], 0.5)
                Oiso[i,1,0] = 0 + min(self.noise[i], 0.5)
                Oiso[i,1,1] = 1 - min(self.noise[i], 0.5)
            
        return Oiso