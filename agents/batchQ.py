# -*- coding: utf-8 -*-
"""
Batch learner for temporal difference Q learning

Should converge to standard temporal difference Q learning for batchsize=1
"""

import numpy as np

def get_softmax_policy(intensity_of_choice):
    """Returns a softmax policy with itensity of choice"""
    beta = intensity_of_choice
    
    def softmax_policy(Qvalues_oa):
        """Returns softmax action probabilites from Qvalues"""
        betaQoa = beta * Qvalues_oa
        betaQoa_ = betaQoa - betaQoa.mean(-1, keepdims=True)
        expQoa = np.exp(betaQoa_)
        assert not np.any(np.isinf(expQoa)), "behavior policy contains infs"
        return expQoa / expQoa.sum(axis=-1, keepdims=True)
    
    return softmax_policy
    
    
def get_epsilongreedy_policy(epsilon):
    """Returns epsilon greedy policy with epsilon"""
    
    def epsilongreedy_policy(Qvalues_oa):
        """Returns softmax action probabilites from Qvalues"""
        
        X = np.zeros_like(Qvalues_oa)
        
        # where are the actions with maximal value?
        maxX = Qvalues_oa == np.max(Qvalues_oa, axis=-1, keepdims=True)
        
        # assign 1-eps probability to max actions
        X += (1-epsilon) * maxX / maxX.sum(axis=-1, keepdims=True)
        
        # assign eps probability to other actions
        othX = np.logical_not(maxX)
        X += epsilon * othX / othX.sum(axis=-1, keepdims=True)
        
        assert np.allclose(X.sum(-1), 1.0)
        
        return X
    
    return epsilongreedy_policy



class batchQ():

    def __init__(self,
                 Qvalues_oa,
                 discount_factor,
                 learning_rate,
                 policy_function,
                 batchsize=1,
                 verbose=0):

        self.alpha = learning_rate       # learning stepsize / rate
        self.gamma = discount_factor       # discout factor

        # value table: gets updateded while acting with the same policy
        self.valQoa = Qvalues_oa.copy()

        # actor table: used for acting, gets updated in learning step
        self.actQoa = Qvalues_oa.copy()

        # policy
        self.policy_func = policy_function
        self.X = self.policy_func(self.actQoa)

        # batch
        self.batchsize = batchsize       
        # GERNEAL AGENT
        self.current_act = None
        self.current_obs = None
        self.next_obs = None
        self.As =  np.arange(self.actQoa.shape[1])

        self.batch_step = 0
        self.total_step = 0
        self.ret = 0

        # batch
        Q, M = self.actQoa.shape
        self.count_oa = np.zeros((Q, M))
        self.count_oao = np.zeros((Q, M, Q))
        self.reward_oa = np.zeros((Q, M))

        self.verbose = verbose


    def estimate_X(self):
        divcount = self.count_oa.sum(-1, keepdims=True).copy()
        divcount[np.where(divcount == 0)] = 1
        return self.count_oa / divcount
       
    def estimate_T(self):
        divcount = self.count_oao.sum(-1, keepdims=True).copy()
        divcount[np.where(divcount == 0)] = 1
        return self.count_oao / divcount
    
    def estimate_Rioa(self):
        divcount = self.count_oa.copy()
        divcount[np.where(self.count_oa == 0)] = 1
        return self.reward_oa / divcount
    
    def estimate_Qioa(self):
        return self.valQoa
    
    def estimate_MaxQioa(self):  
        return np.dot(self.estimate_T(), self.valQoa.max(-1))
    
    def estimate_TDioa(self):               
        TDe = ((1-self.gamma) * self.estimate_Rioa()
               + self.gamma * self.estimate_MaxQioa()
               - self.actQoa)
        
        return TDe


    def interact(self, observation, reward):

        if reward is not None:
            self.batchstore(reward, observation)

            if self.batch_step == self.batchsize:
                self.batchlearn()

        # returns with the next action
        action = self.act(observation)
        return action
    
    def act(self, observation):
        """
        Choose action for given observation with boltzmanm probabilites.

        Parameters
        ----------
        observation : int
            the agent's observation of the state of the environment
        """
        # check for consistency while interaction with the environment
        assert (self.next_obs == observation or self.next_obs is None),\
            "Agent's observation has changed inconsistently"

        action = np.random.choice(self.As, p=self.X[observation])

        self.current_obs = observation
        self.current_act = action

        return action


    def batchstore(self, reward, next_obs):
        """
        store experience inside batch
        """
        self.count_oa[self.current_obs, self.current_act] += 1
        self.count_oao[self.current_obs, self.current_act, next_obs] += 1
        self.reward_oa[self.current_obs, self.current_act] += reward
        
        # updating the value table, estiamting the current state-action values
        self.valQoa[self.current_obs, self.current_act]\
            += self.alpha * ((1-self.gamma) * reward\
            + self.gamma * np.dot(self.X[next_obs], self.valQoa[next_obs])\
            - self.valQoa[self.current_obs, self.current_act])

        self.next_obs = next_obs  # just for consistency checking
        
        self.ret = (1-self.gamma)*reward + self.gamma * self.ret
        self.batch_step += 1
        self.total_step += 1


    def batchlearn(self, final_state=False):
        """
        Use batch to update acting observation-action values
        """
        # what to do anyway with finals states in batch learning ???

        TDe = self.estimate_TDioa()     
        self.actQoa += self.alpha * TDe
        self.X = self.policy_func(self.actQoa)
        
        
        print("\r>> batchlearn at {}".format(self.total_step), 
              end='', flush=True)  if self.verbose else None
        print() if self.verbose else None
        print('actQ:\n', self.actQoa) if self.verbose else None
        print('TDe:\n', TDe) if self.verbose else None
        print() if self.verbose else None


        #re-init
        self.count_oa.fill(0)
        self.count_oao.fill(0)
        self.reward_oa.fill(0)
        # self.valQoa = self.actQoa.copy() # may not be needed
        self.batch_step = 0


    def get_actQ(self):
        return self.actQoa.copy()