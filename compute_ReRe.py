# %% imports
import os
import numpy as np
import pandas as pd

from environments.Env_ReRe import ReRe as ENV
from agents.deterministic import detQ

from utils.valueiteration import value_iteration, average_reward,\
    avgreward_value_iteration

#%% 

def unique_pols(lastXs, NrObs, M):
    uniX = np.unique(np.array(lastXs).round(1), axis=0)
    mults = []
    
    for i in range(len(uniX)):
        m = np.where(np.where(np.array(lastXs).round(1) == uniX[i],1,0)
             .sum(axis=-1).sum(axis=-1).flatten() == NrObs*M, 1, 0).sum()
        mults.append(m)
        
    return uniX.tolist(), mults

def compute_trajectory(agents, Xinit, Tmax=10000, fpepsilon=0.000001):

    X = Xinit.copy()
    Xtraj = []
    Rtraj = []
    fixpreached = False
    t = 0
    while not fixpreached and t < Tmax:
        Xtraj.append(X)

        Rio = agents.obtain_Rio(X)
        pio = agents.obtain_obsdist(X)
        Rtraj.append(np.einsum(pio, [0,1], Rio, [0, 1], [0]))

        Xnew = agents.TDstep(X)
        fixpreached = np.linalg.norm(Xnew - X) < fpepsilon
        X = Xnew
        t += 1

    return Xtraj, Rtraj, fixpreached, X

def run(growthrate = 0.8,
        capacity = 5,
        deltaE=0.2, 
        sig=0.5,
        observation = None,
        samplesize = 100,
        Tmax = 750,
        datafolder=None):
    
    env = ENV(r=growthrate, C=capacity, obs=observation, deltaE=deltaE, sig=sig)
    T = env.TransitionTensor().astype(float); # print(T.sum(-1))
    R = env.RewardTensor().astype(float)
    O = env.ObservationTensor(); # print(O.sum(-1))
    
    #%% init agent
    alpha = 0.02
    beta = 250
    gamma = 0.9
    agents = detQ(T, R, O, alpha, beta, gamma, opteinsum=False)
    
    # reporting
    print("Observation:", env.obs)
    gd = env._growth_dict()
    print("Growth:", {k: np.round(gd[k], 3) for k in gd})
    av = env._action_values()
    print("Action Vals:", np.array(av).round(3))
    
    # file handling
    fn=f'ReRe_C{env.C}_r{env.r}_dE{env.dE}_sig{env.sig}_Obs{env.obs}_SaSi{samplesize}_Tmax{Tmax}'+\
        f'alpha{alpha}_beta{beta}_gamma{gamma}'

    storage = os.path.expanduser(datafolder) if datafolder is not None else ''
    
    print(storage+fn)
    
    # check if file exisits
    dataexists = os.path.isfile(storage+fn+".csv")
    
    if dataexists:
        print("Data exists already")
    else:
        print("Computing")
  
        lens = []; rews = []; lastXs = []
        for _ in range(samplesize):
            print(".",  end=" ")
            Xinit = agents.random_behavior()
            xt, rt, fpr, lastX = compute_trajectory(agents, Xinit, Tmax,
                                                    fpepsilon=0.0001)
            lens.append(len(rt))
            rews.append(rt[-1])
            lastXs.append(lastX)
        print()
        
        # # find optimal reward
        Xop = value_iteration(R, T, gamma)
        Rop = average_reward(Xop, R, T, gamma)
        
        AvgXop = avgreward_value_iteration(R, T)
        AvgRop = average_reward(AvgXop, R, T)
        
        # find unique policies
        uniX, Xmults = unique_pols(lastXs, NrObs=len(observation), M=3)
    
        # who converged?
        allconverged = not (np.array(lens) == Tmax).any()
        
        # reporting
        data = pd.Series(data={"AVG LenTraj": np.round(np.mean(lens), 3),
                               "STD LenTraj": np.round(np.std(lens), 3),
                               "Rop": np.round(Rop, 3),
                               "AvgRop": np.round(AvgRop, 3),
                               "AVG Rew": np.round(np.mean(rews), 3),
                               "STD Rew": np.round(np.std(rews), 3),
                               "NrX": len(Xmults),
                               "AllConverged": allconverged})
        print()
        print(data)
        
        Xdat = pd.DataFrame(dict(X=uniX, mult=Xmults))
        print()
        print(Xdat)
        
        # saving plot
        if datafolder is not None:
            # saving data
            data.to_csv(storage+fn+".csv", header=False)
            Xdat.to_csv(storage+fn+"_Xdat.csv")
    
        return lastXs
#%% exec

# # parameters
# datafolder = "./"
# gr = 0.8
# dE = 0.2
# sig = 0.5
# Tmax = 750
# SaSi = 100
# print("--- COPMUTE Renewable Resource ---")
# print("gr  :", gr)
# print("dE  :", dE)
# print("sig :", sig)
# print("Tmax:", Tmax)
# print("SaSi:", SaSi)
# print("----------------------------------")
# print()

# for cap in range(4,8):      
# for cap in range(4,10):      
#     allobs = getAllObservations(cap)
#     print()
#     print("C A P A C I T Y ", cap, f" - has {len(allobs)} observations")
#     print()
#     for i, obs in enumerate(reversed(allobs)):
#         print("OBSERVATION Nr", i)
#         run(growthrate=gr, capacity=cap, deltaE=dE, sig=sig, observation=obs,
#             Tmax=Tmax, samplesize=SaSi)
#         print()
#         plt.close()
