

import numpy as np
from .Env_Null import NullEnv

states = [(0, 0),  # state 0
            (0, 1),  # state 1
            (0, 2),  # state 2
            (1, 0),  # state 3
            (1, 2),  # state 4
            (2, 0),  # state 5
            (2, 1),  # state 6
            (2, 2),  # state 7
            (3, 0),  # state 8
            (3, 1),  # state 9: the penalty state
            (3, 2)   # state 10: the goal state
            ]

actions = [(+1, 0),  # action: move east
            (-1, 0),  # action: move west
            (0, +1),  # action: move north
            (0, -1),  # action: move south
            ]
    
class ParrRusselGridWorld(NullEnv):
    
    def __init__(self, action_error_rate=0.2, reward_step_penalty=0.04,
                 observe="partial"):
        """
        Parr and Russel's Grid World
        
        STATES
        11 states in a 4 by 3 grid with a single obstacle at (2,2). The state of
        the environment is determined by the grid square occupied by the agent
        - 0: 0,0
        - 1: 0,1
        - 2: 0,2
        - 3: 1,0
        - 4: 1,2
        - 5: 2,0
        - 6: 2,1
        - 7: 2,2
        - 8: 3,0
        - 9: 3,1 (the penality state)
        -10: 3,2 (the goal state)
        
        ACTIONS
        The agent can choose one of 4 actions: move north, move south, move
        east, and move west. State transitions are stochastic with the agent
        moving in the desired direction 80% of the time and slipping to either
        side 10% of the time. (default parameters)
        If such a movement is obstructed by a wall, then the agent will stay
        put instead
        - 0: move north
        - 1: move south
        - 2: move east
        - 3: move west
        
        OBSERVATIONS
        The agent can only observe if there is a wall to its immediate east or
        west. There are 4 possible observations corresponding to the
        combinations of left and right obstacles plus two observations for the
        goal and penalty states yielding a total of 6 observations. 
        Observations are deterministic.
        - 0: wall west
        - 1: wall east
        - 2: no wall
        - 3: wall east and west
        - 4: Penalty state
        - 6: Goal states
         
        REWARDS
        There is a goal state in the upper right corner with a penalty state
        directly below the goal state. The agent receives a reward of −0.04 for
        every action which does not lead to the goal or penalty state. The
        agent receives a reward of +1 for any action leading to the goal state
        and a reward of −1 for any action leading to the penalty state
        (default parameters)

        Introduced in
        Parr & Russel, 1995 - Approximating optimal policies for partially
        observable stochastic domains.

        and used in e.g.,
        -   Williams & Singh, 1999 -  Experimental Results on Learning
            Stochastic Memoryless Policies for Partially Observable Markov
            Decision Processes
        -   Loch & Singh, 1998 - Using Eligibility Traces to Find the Best
            Memoryless Policy in Partially Observable Markov Decision Processes
        """       
        # parameters
        assert action_error_rate >= 0 and action_error_rate <= 1.0
        self.action_error_rate = action_error_rate
        assert reward_step_penalty >= 0 and reward_step_penalty <= 1.0
        self.reward_step_penalty = reward_step_penalty
        assert observe in ["full", "partial"]
        self.observe = observe
        
        self.N = 1   # Number of agents
        self.M = 4   # Number of actions
        self.Z = 11  # Number of states
        self.Q = 6 if observe == "partial" else self.Z  # Number of observations
        
        self.T = self.TransitionTensor()
        self.R = self.RewardTensor()
        self.O = self.ObservationTensor()
        
        
    def TransitionTensor(self):
        """Get the Transition Tensor."""
        dim = np.concatenate(([self.Z],
                              [self.M for _ in range(self.N)],
                              [self.Z]))
        Tsas = np.ones(dim) * (-1)

        for index, _ in np.ndenumerate(Tsas):
            Tsas[index] = self._transition_probability(index[0],
                                                       index[1:-1],
                                                       index[-1])
        return Tsas
    
    def _transition_probability(self, s, jA, sprim):

        # translating states and actions into coordinates
        x, y = states[s]
        xp, yp = states[sprim]
        dx, dy = actions[jA[0]]
        
        # actions due to systematic errors
        dxe1, dye1 =  int(not dx),  int(not dy)
        dxe2, dye2 = -int(not dx), -int(not dy)
        
        # # goal and penalty states
        if (s==9 or s==10):
            if (sprim==9 or sprim==10):
                return 0
            else:
                return 1/9
        else:
            # movement
            if x+dx==xp and y+dy==yp:
                return 1-self.action_error_rate
            elif x+dxe1==xp and y+dye1==yp:
                return self.action_error_rate / 2
            elif x+dxe2==xp and y+dye2==yp:
                return self.action_error_rate / 2
            # no movement possible -> agent stays
            elif x==xp and y==yp:
                # check which movements are not possible and add transition
                # probabilities
                p = 0
                if (x+dx, y+dy) not in states:
                    p += 1-self.action_error_rate
                if (x+dxe1, y+dye1) not in states:
                    p += self.action_error_rate/2
                if (x+dxe2, y+dye2) not in states:
                    p += self.action_error_rate/2
                return p    
            else:  # the rest
                return 0


    def RewardTensor(self):
        """Get the Reward Tensor R[i,s,a1,...,aN,s']."""
        dim = np.concatenate(([self.N],
                              [self.Z],
                              [self.M for _ in range(self.N)],
                              [self.Z]))
        Risas = np.zeros(dim)

        for index, _ in np.ndenumerate(Risas):
            Risas[index] = self._reward(index[0], index[1], index[2:-1],
                                        index[-1])
        return Risas

    def _reward(self, i, s, jA, sprim):
        # if sprim == 9:  # penalty state
        #     reward = -1
        # elif sprim == 10:  # goal state
        #     reward = 1
        # elif s==9 or s==10:
        #     reward = 0
        if s == 9:
            reward = -1
        elif s == 10:
            reward = 1
        else:
            reward = -self.reward_step_penalty

        return reward 
    
    def ObservationTensor(self):
        """Get the Observation Tensor O[i,s,o]."""
        Oiso = np.ones((self.N, self.Z, self.Q))
        
        for index, _ in np.ndenumerate(Oiso):
            Oiso[index] = self._observation(index[0], index[1], index[2])
        return Oiso

    def _observation(self, i, s, o):
        """
        The agent can only observe if there is a wall to its immediate east or
        west. There are 4 possible observations corresponding to the
        combinations of left and right obstacles plus two observations for the
        goal and penalty states yielding a total of 6 observations. 
        Observations are deterministic.
        - 0: wall west
        - 1: wall east
        - 2: no wall
        - 3: wall east and west
        - 4: Penalty state
        - 5: Goal states
        """      
        
        if self.observe == "partial":
            s_wallwest = [states.index(xy) for xy in [(0,0), (0,2), (2,1)]]
            s_walleast = [states.index((3,0))]
            s_nowall = [states.index(xy) for xy in [(1,0), (2,0), (1,2), (2,2)]]
            s_walleastwest = [states.index((0,1))]

            if s in s_wallwest and o==0:
                return 1.0
            elif s in s_walleast and o==1:
                return 1.0
            elif s in s_nowall and o==2:
                return 1.0
            elif s in s_walleastwest and o==3:
                return 1.0
            elif s==9 and o==4:
                return 1.0
            elif s==10 and o==5:
                return 1.0
            else:
                return 0.0
        elif self.observe == "full":
            
            if s == o:
                return 1.0
            else:
                return 0.0
  
        # Wall west < (0,0), (0,2), (2,1)
        