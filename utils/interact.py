# -*- coding: utf-8 -*-
"""
Interact.py - let agent(s) interact with environment
"""
# imports
import time
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
# from utils.quiverplot import plot_quiver, gridded_policies

def create_name(env, sampsize, learner, detAgents=None):
    envsamps = f"{env.__class__.__name__}_{sampsize}Xtrajs_"
    learName = f"{learner.__class__.__name__}_"
    if hasattr(learner, "batchsize"):
        learName += f"bsize{learner.batchsize}_"
    params = f"alpha{learner.alpha}_beta{learner.beta}_gamma{learner.gamma}"
    
    if detAgents is not None:
        detName = f"{detAgents.__class__.__name__}_"
        return envsamps + learName + detName + params
    
    else: 
        return envsamps + learName + params


# TODO: improve stopping criterion
def interface(agents, envrionment, steps):
    """
    Agents-Environment Interface. Let the agents interact.
    
    Parameters
    ----------
    agents : iterable of agents
    envrionment : envrionment
    steps : int (Number of interaction steps)
    """
    def allagentinteract(obs, rews):
        actions = []
        for ag in range(envrionment.N):
            actions.append(agents[ag].interact(obs[ag], rews[ag]))
        return np.array(actions)
   
    # INTERACTION
    obs = envrionment.observation()
    actions = allagentinteract(obs, np.array([None]).repeat(envrionment.N))
    
    # data
    Xtraj = [[agent.X for agent in agents]]  # policies
    Rtraj = [[0 for _ in agents]]  # rewards
    Gtraj = [[agent.ret for agent in agents]]  # returns / gains
    Otraj = [obs]  # observations
    Straj = [envrionment.state]  # env / state
    Atraj = [actions] # agents
    if hasattr(agents[0], 'actQoa'):
        Qtraj = [[agent.get_actQ() for agent in agents]]
    
    # print(actions)  # TODO: verbosity 
    for _ in range(steps):
        obs, rews, done, info = envrionment.step(actions)
        actions = allagentinteract(obs, rews)

        # data
        Xtraj.append([agent.X for agent in agents])
        Rtraj.append(rews)
        Gtraj.append([agent.ret for agent in agents])
        Otraj.append(obs)
        Straj.append(envrionment.state)
        Atraj.append(actions)
        if hasattr(agents[0], 'actQoa'):
            Qtraj.append([agent.get_actQ() for agent in agents])

    data = dict(policies=Xtraj,
                rewards=Rtraj,
                returns=Gtraj,
                observations=Otraj,
                states=Straj,
                actions=Atraj)
    
    if hasattr(agents[0], 'actQoa'):
        data['OAvalues'] = Qtraj

    return data
    
    # # data
    # return Xtraj  # , Rtraj



def compute_Xtrajs(learners, env, length, sampsize, savepath=None,
                   includeXinSavePath=False, fidadd=''):
    """
    Compute and save /load X trajectories of a batch learning algorithm.

    Parameters
    ----------
    learners : learning algo
        Batch reinforcement learning algorithm
    env : env
        Environment.
    length : int
        Number of total interaction steps with the envrionment.
    sampsize : int
        Sample size of identical repetitions.
    savepath : str
        Location of data storage path.
    includeXinSavePath : bool (default: False)
        If True, include initial policy in filename

    Returns
    -------
    Xtrajs : list
        List of behavior (X) trajectories. Each list item contains the X-
        trajectory of a sample run.
    """
    # check whether learners is single instance or iterable
    if hasattr(learners, '__iter__'):
        assert len(learners) == env.N
    else:
        learners = [deepcopy(learners) for _ in range(env.N)]
    learner = learners[0]
    for l in learners: # check that all learners are of the same class
        # not required in principle, but current file nameing suggests that
        assert learner.__class__.__name__ == l.__class__.__name__

    if savepath is not None:
        if includeXinSavePath:
            Xstr = "_X"
            for l in learners:
                Xstr += str(l.X.flatten())
        else:
            Xstr = ''

        fname = savepath + create_name(env, sampsize, learner) + Xstr +\
            fidadd + ".npz"
            
        save = True
    else:
        save = False
    
    try:
        Xtrajs = np.load(fname)['dat']
        print("Loading")
    except:
        print("Computing: Total number of timesteps: ", length)
        Xtrajs = []
    
        start_time = time.time()
        for i in range(sampsize):
            print('  sample', i)
            #listoflearners = [deepcopy(learner) for _ in range(env.N)]
            listoflearners = deepcopy(learners)
            trajectory = interface(listoflearners, env, length)
            Xtrajs.append(trajectory['policies'])
        if save: np.savez_compressed(fname, dat=np.array(Xtrajs))
    
        now = time.time()
        print(f"--- {now - start_time} seconds ---")
        if sampsize > 0:
            print(f" == {(now - start_time)/sampsize} seconds/sample == ")

    return Xtrajs



def compute_detXtraj(agents, Xinit=None, Tmax=100, EpsMin=None,
                     verbose=False):
    """
    Computes a learning trajectory from a detAgent
    
    agents : detAgent object
    Xinit : the inital behaviour (if None given use random policy)
    Tmax : the maximum number of iteration steps (default: 100)
    EpsMin : to determine if a fix point is reached (default: None) 
    return_rewards : wheter to return rewards (default: False)
    
    ---
    TODO: compare and unify with quiver > tracetory
    """
    Xtraj, Rtraj = [], []
    fixpreached = False
    if Xinit is not None:
        X = Xinit.copy()
    else:
        X = agents.random_behavior()

    t = 0
    # to have two datapoints if Xinit is already fixed point
    Ris = agents.obtain_Ris(X)
    δ = agents.obtain_statedist(X)       
    Rtraj.append(np.sum(δ.T * Ris, axis=1))
    Xtraj.append(X)
    
    while not fixpreached and t < Tmax:
        if verbose:
            print(f" [computing trajectory] step {t}")
        Ris = agents.obtain_Ris(X)
        δ = agents.obtain_statedist(X)       
        Rtraj.append(np.sum(δ.T * Ris, axis=1))
        Xtraj.append(X)

        Xnew = agents.TDstep(X)
        if EpsMin is not None:
            fixpreached = np.linalg.norm(Xnew - X) < EpsMin
        X = Xnew
        t += 1
    if verbose:
        print(f" [trajectory computed]")


    return np.array(Xtraj), np.array(Rtraj), fixpreached


def compute_detRL_policy_trajectory(agents, Xinit=None, Tmax=100, EpsMin=None,
                                    verbose=False):
    print(['DEPRECATED: USE `compute_detRL_trajectory` INSTEAD'])
    return compute_detRL_trajectory(agents, Xinit, Tmax, EpsMin, verbose)
            
            
def compute_detRL_trajectory(agents, Xinit=None, Tmax=100, EpsMin=None,
                                    verbose=False):
    """
    Computes a learning trajectory from a detRL agent
    
    agents : detRL Agent object
    Xinit : the inital behaviour (if None given use random policy)
    Tmax : the maximum number of iteration steps (default: 100)
    EpsMin : to determine if a fix point is reached (default: None) 
    
    ---
    TODO: compare and unify with quiver > tracetory
    """
    Xtraj = []
    fixpreached = False
    if Xinit is not None:
        X = Xinit.copy()
    else:
        X = agents.random_behavior()

    t = 0
    # Xtraj.append(X) # to have two datapoints if Xinit is already fixed point
    while not fixpreached and t < Tmax:
        print(f" [computing trajectory] step {t}") if verbose else None 
        Xtraj.append(X)

        Xnew = agents.TDstep(X)
        if EpsMin is not None:
            fixpreached = np.linalg.norm(Xnew - X) < EpsMin
        X = Xnew
        t += 1
    print(f" [trajectory computed]") if verbose else None
    
    return np.array(Xtraj), fixpreached


        
def detRL_reward_trajectory_from_policies(agents, Xtraj):
    """
    Convert policy trajectory to reward trajectory
    """
    Rtraj = np.array([np.einsum(agents.obtain_statedist(X), [1],
                                agents.obtain_Ris(X), [0,1], [0])
                      for X in Xtraj])
    return Rtraj

# = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = 
#   VIZUALISATIONS
# = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = - = 

def visualize_BehaviorSpace(
        detAgents, Xinits=[], detXtrajs=None, Xtrajs=None, PlotFor="Z",
        DiffType="TDe", pAs = np.linspace(0.01, 0.99, 8), cmap='viridis',
        axes=None,
        detkwargs=dict(color='darkred', lw=3.0, ms=5.5, alpha=0.9),
        algkwargs=dict(color="deepskyblue", ms=0.1, alpha=0.4, zorder=1)):
    
    # quiver
    axes = plot_quiver(detAgents, pAs=pAs, plot_for=PlotFor, sf=0.5,
                       difftype=DiffType, cmap=cmap, axes=axes)
    
    # trajectories
    if PlotFor == "C":
        ixx = lambda j: (slice(-1), 0, j, 0)
        ixy = lambda j: (slice(-1), 1, j, 0)
    elif PlotFor == "N":
        ixx = lambda j: (slice(-1), j, 0, 0)
        ixy = lambda j: (slice(-1), j, 1, 0)
    else:
        assert False, "'PlotFor' must be either 'C' or 'N'"

    if detXtrajs is not None:
        for detXtraj in detXtrajs:
            for j, ax in enumerate(axes):
                ax.plot(np.array(detXtraj)[ixx(j)], np.array(detXtraj)[ixy(j)],
                        "--", **detkwargs)
        
    if Xtrajs is not None:
        for Xtraj in Xtrajs:
            for j, ax in enumerate(axes):
                ax.plot(np.array(Xtraj)[ixx(j)], np.array(Xtraj)[ixy(j)],
                        "-", **algkwargs)

    return axes

# =============================================================================
# =============================================================================
#       CONGERGENCE TERMS
# =============================================================================
# =============================================================================

# helper function
def allagentinteract(learners, env, obs, rews):
    actions = []
    for ag in range(env.N):
        actions.append(learners[ag].interact(obs[ag], rews[ag]))
    return np.array(actions)

def get_convergence_terms(
    detAs, algAs, env, Xinit, testtimes, sampsize, savepath=None, fidadd=None,
    useTDproxy=False, useTDnorm=True):
    
    if savepath is not None:
        fn = lambda x : "_" + x.__class__.__name__
        fid = "CONV" + fn(env) + fn(detAs) + fn(algAs[0]) + f"_samps{sampsize}"
        fname = savepath + fid + fidadd + ".npz"
        save = True
    else:
        save = False

    try:
        dat = np.load(fname)
        convterms = dict(zip((k for k in dat), (dat[k] for k in dat)))
        print("Loading")
    except:
        print("Computing")
        convterms = compute_convergence_terms( detAs, algAs, env, Xinit,
            testtimes, sampsize, useTDproxy, useTDnorm)
        if save:
            np.savez_compressed(fname, **convterms)
        
    return convterms

def compute_convergence_terms(detAs, algAs, env, Xinit, testtimes, sampsize,
                              useTDproxy=False, useTDnorm=True):
    assert len(algAs) == env.N

    def _diff(esti, tru):
        #return np.abs( (esti - tru) / tru.sum() ).mean()
        return np.abs( (esti - tru) / np.abs(tru).mean() ).mean()

    # terms
    DactX_ss = []
    DXioa_ss = []
    DRioa_ss = []
    DTioao_ss = []
    DQioa_ss = []
    DMaxQioa_ss = []
    DTDioa_ss = []
        
    for samp in range(sampsize):
        print("[compute_convergence_terms] Sample: ", samp)
        j = 0 # testtimes iterator
        learners = deepcopy(algAs)
        X = Xinit.copy()
        
        # containers
        DactX_s = []
        DRioa_s = []
        DXioa_s = []
        DTioao_s = []
        DQioa_s = []
        DMaxQioa_s = []
        DTDioa_s = []
            
        # INTERACTION
        actions = allagentinteract(learners, env, env.observation(),
                                   np.array([None]).repeat(env.N))
        for i in range(testtimes.max()):
            obs, rews = env.step(actions)
            actions = allagentinteract(learners, env, obs, rews)
            
            if i % algAs[0].batchsize == 0:  # learning takes place
                if i>0:
                    X = detAs.TDstep(X); print("det learn at ", i)
                # updateing deterministic results
                actX = np.array([1/learners[i].beta *  np.log(X[i])
                                 for i in range(len(learners))])
                Tioao = detAs.obtain_Tioao(X)
                Rioa = detAs.obtain_Rioa(X)
                Qioa = detAs.obtain_Qioa(X)
                MaxQioa = detAs.obtain_MaxQioa(X)
                
                TDioa = detAs.TDerror(X, norm=False)

                if useTDproxy:
                    TDioa = np.array([(1-l.gamma) * Rioa +
                                      l.gamma  * MaxQioa                            
                                      for l in learners])
                

                if useTDnorm:
                    TDioa = TDioa - TDioa.mean(axis=-1, keepdims=True)
                    actX = actX - actX.mean(axis=-1, keepdims=True)

            if i == testtimes[j]:
                eXioa = np.array([l.estimate_X() for l in learners])
                eActX = np.array([l.actQoa for l in learners])
                eTioao = np.array([l.estimate_T() for l in learners])
                eRioa = np.array([l.estimate_Rioa() for l in learners])
                eQioa = np.array([l.estimate_Qioa() for l in learners])
                eMaxQioa = np.array([l.estimate_MaxQioa() for l in learners])
                eTDioa = np.array([l.estimate_TDioa() for l in learners])

                if useTDproxy:
                    eTDioa = np.array([(1-l.gamma) * eRioa +
                                       l.gamma  * eMaxQioa
                                       for l in learners])

                if useTDnorm:
                    eTDioa = eTDioa - eTDioa.mean(axis=-1, keepdims=True)
                    eActX = eActX - eActX.mean(axis=-1, keepdims=True)


                DactX_s.append(_diff(eActX, actX))
                DXioa_s.append(_diff(eXioa, X))
                DTioao_s.append(_diff(eTioao, Tioao))
                DRioa_s.append(_diff(eRioa, Rioa))
                DQioa_s.append(_diff(eQioa, Qioa))
                DMaxQioa_s.append(_diff(eMaxQioa, MaxQioa))
                DTDioa_s.append(_diff(eTDioa, TDioa))
                j += 1
    
        DactX_ss.append(DactX_s)
        DXioa_ss.append(DXioa_s) 
        DTioao_ss.append(DTioao_s)
        DRioa_ss.append(DRioa_s)
        DQioa_ss.append(DQioa_s)
        DMaxQioa_ss.append(DMaxQioa_s)
        DTDioa_ss.append(DTDioa_s)
        
    return dict(DXioa=np.array(DXioa_ss),
                DactX=np.array(DactX_ss),
                DRioa=np.array(DRioa_ss),
                DTioao=np.array(DTioao_ss),
                DQioa=np.array(DQioa_ss),
                DMaxQioa=np.array(DMaxQioa_ss),
                DTDioa=np.array(DTDioa_ss),
                testtimes=testtimes)

def plot_convterms(convterms, keys, colors, labels=None, title="", ax=None):
      
    if ax is None:
        fig, ax = plt.subplots()

    if labels is None:
        labels = keys

    xvs = convterms['testtimes'][:-1]
    for i, k in enumerate(keys):
    
        mean = convterms[k].mean(0)
        std = convterms[k].std(0)
        ax.plot(xvs, mean, color=colors[i])
        ax.fill_between(xvs, mean-0.5*std, mean+0.5*std, alpha=0.4,
                         color=colors[i], label=labels[i])
    
    ax.set_title(title)
    plt.legend()

    return ax