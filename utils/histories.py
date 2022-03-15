"""
Embed an environment into a more complex representation
of state/observation-action histories
"""
# TODO: take of final states
import numpy as np
import itertools as it

# For representation
def hSset(env, h):
    hmax = max(h)
    
    hists = []
    for hist in StateActHistsIx(env, h):
        hrep = ''
        # go through all steps of the history
        for step in range(0, hmax):
            # first: all actions
            for i, n in enumerate(range(step*(env.N+1), step*(env.N+1)+env.N)):
                hrep += env.Aset[i][hist[n]] if hist[n]!="." else ''
                hrep += ','
            # second: append state
            hrep += env.Sset[hist[n+1]] if hist[n+1]!="." else ''
            hrep += '|'
        hists.append(hrep)
    
    return hists

def hOset(env, h):
    hmax = max(h)
    
    all_hists = []
    for agent in range(env.N):
        hists = []
        for hist in ObsActHistsIx(env, h):
            hrep = ''
            # go through all steps of the history
            for step in range(0, hmax):
                # first: all actions
                for i, n in enumerate(range(step*(env.N+1), step*(env.N+1)+env.N)):
                    hrep += env.Aset[i][hist[n]] if hist[n]!="." else ''
                    hrep += ','
                # second: append observation
                    hrep += env.Oset[agent][hist[n+1]] if hist[n+1]!="." else ''
                    hrep += '|'
            hists.append(hrep)
        all_hists.append(hists)
    
    return all_hists
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =        
#   State-Observation-Action Histories
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 

def StateActHistsIx(env, h):
    """
    Returns all state-action histories of `env`.
    `h` specifies the type of history.
        `h` must be an iterable of length 1+N (where N = Nr. of Agents)
        The first element of `h` specifies the length of the state-history
        Subsequent elements specify the length of the respective action-history
        
    Example
    -------
    # for a two-agent environment
    StateActHistsIx(env, h=(2,0,1))
    """
    
    # get all hists
    Hists = _get_all_histories(env, h)

    # Remove squences that are not possible
    PossibleHists = Hists.copy()
    for hist in Hists:
        if _hist_contains_NotPossibleTrans(env, hist):
            PossibleHists.remove(hist)
    return PossibleHists

def _get_all_histories(env, h, attr='Z'):
    assert len(h) == env.N+1
    assert np.all(np.array(h)>=0)

    hiter = []  # history iterables 
    # go through the maximum history length
    for l in reversed(range(max(h))):
        # first: actions
        # go through all agents
        for n in range(env.N):
            # if 
            if l<h[1+n]:
                hiter.append(list(range(env.M)))
            else:
                hiter.append('.')

        # second: state
        # if specified hist-length is larger than current length
        if h[0] > l:
            # append hiter with range of states
            hiter.append(list(range(getattr(env, attr))))
        else:
            # if not: append dummy sign
            hiter.append('.')

    hists = list(it.product(*hiter)) 
    return hists

def _hist_contains_NotPossibleTrans(env, hist):
    assert len(hist)%(env.N+1) == 0
    maxh = int(len(hist) / (env.N+1)) # max history length

    contains = False
    # go through history from past to present
    s='.'
    for step in range(0, maxh):
        jA = hist[step*(env.N+1):step*(env.N+1)+env.N]
        s_ = hist[step*(env.N+1)+env.N]

        # construcing index for transition tensor
        ix = [s] if s!='.' else [slice(env.Z)]
        ix+= [jA[n] if jA[n]!='.' else slice(env.M) for n in range(env.N)]
        ix+= [s_] if s_!='.' else [slice(env.Z)]

        # check wheter there is possibility for current s,jA,s' tripple
        probability = np.sum(env.T[tuple(ix)])

        if probability==0:
            contains = True
            break
        else:
            # set new state s to s_
            s = s_
    return contains

def histSjA_TransitionTensor(env, h):
    """
    Returns Transition Tensor of `env` with state-action history `h`[iterable]
    """
    hmax = max(h)

    def _transition_possible(hist, hist_):
        hi=hist[env.N+1:]; 
        hi_=hist_[:-(env.N+1)]
        possible = []
        for k in range((hmax-1)*(env.N+1)):
            poss = (hi[k]=='.') or (hi_[k]=='.') or (hi[k]==hi_[k])
            possible.append(poss)
        return np.all(possible)

    Hists = StateActHistsIx(env, h)

    Zh = len(Hists)
    Th_dims = list(env.T.shape)
    Th_dims[0] = Zh
    Th_dims[-1] = Zh
    Th = np.zeros(Th_dims)

    for i, hist in enumerate(Hists):
        for j, hist_ in enumerate(Hists):
            # Is the transition possible?
            possible = _transition_possible(hist, hist_)  
            # Get indices
            hix, ix = _transition_ix(env, h, i, hist, j, hist_)
        
            Th[hix] = possible*env.T[ix]

    return Th

def _transition_ix(env, h, i, hist, j, hist_):
    hmax = max(h)

    s = hist[-1]
    jA = hist_[(hmax-1)*(env.N+1):(hmax-1)*(env.N+1)+env.N]
    s_ = hist_[-1]

    # construcing index for transition tensor
    jAx = [jA[n] if jA[n]!='.' else slice(env.M) for n in range(env.N)]

    # for original tensor
    ix = [s] if s!='.' else [slice(env.Z)]
    ix += jAx
    ix+= [s_] if s_!='.' else [slice(env.Z)]

    # for history tensor
    hix = [i]+jAx+[j]

    return tuple(hix), tuple(ix)
    
def ObsActHistsIx(env, h):
    """
    Returns all obs-action histories of `env`.
    `h` specifies the type of history.
        `h` must be an iterable of length 1+N (where N = Nr. of Agents)
        The first element of `h` specifies the length of the obs-history
        Subsequent elements specify the length of the respective action-history
        
    Note: Here only partial observability regarding the envrionmental state
    applies. Additional partial observability regarding action is treated 
    seperatly.
    """
    
    SAhists = StateActHistsIx(env, h=h)
    OAhists = _get_all_histories(env, h=h, attr='Q')

    hmax=max(h)  # the maximum history length
    l = (env.N+1)*hmax  # length of a single history representation

    # Remove squences that are not possible to observe
    # for all agents
    # check all ohist elements
    PossibleOAHists = OAhists.copy()
    for oahist in OAhists:
        # whether they are observable by agent i
        observable = np.zeros(env.N)
        # go through all shist elements
        for sahist in SAhists:
            # check wheter action profile fits
            sAs = [list(sahist[k:k+env.N]) for k in range(0, l, env.N+1)]
            oAs = [list(oahist[k:k+env.N]) for k in range(0, l, env.N+1)]
            if sAs == oAs:
                # and then check whether oahist can be observed from sahist
                observable += np.prod([env.O[:, sahist[k], oahist[k]]
                                    for k in range((env.N+1)*(hmax-h[0])+env.N,
                                                   l, env.N+1)], axis=0)
        # if oahist can't be observed by any agent
        if np.allclose(observable, 0.0):
            # remove ohist from ObsHists
            PossibleOAHists.remove(oahist)
    return PossibleOAHists
    
def histSjA_ObservationTensor(env, h):
    """
    Returns Observation Tensor of `env` with state-action history `h`[iterable]
    """
    hmax=max(h)  # the maximum history length
    l = (env.N+1)*hmax  # length of a single history representation

    SAhists = StateActHistsIx(env, h=h)
    OAhists = ObsActHistsIx(env, h=h)

    Qh = len(OAhists)
    Zh = len(SAhists)
    Oh = np.zeros((env.N, Zh, Qh))

    # go through each sh oh pair
    for i, sahist in enumerate(SAhists):
        for j, oahist in enumerate(OAhists):
            # check wheter action profile fits
            sAs = [list(sahist[k:k+env.N]) for k in range(0, l, env.N+1)]
            oAs = [list(oahist[k:k+env.N]) for k in range(0, l, env.N+1)]
            if sAs == oAs:
                Oh[:, i, j] = np.prod([env.O[:, sahist[k], oahist[k]]
                                for k in range((env.N+1)*(hmax-h[0])+env.N,
                                                l, env.N+1)], axis=0)
    return Oh           

def histSjA_RewardTensor(env, h):
    """
    Returns Reward Tensor of `env` with state-action history `h`[iterable]
    """
    hmax=max(h)  # the maximum history length
    l = (env.N+1)*hmax  # length of a single history representation
                                    
    SAHists = StateActHistsIx(env, h)

    # dimension for history reward tensor
    Zh = len(SAHists)
    dims = list(env.R.shape)
    dims[1] = Zh
    dims[-1] = Zh

    Rh = np.zeros(dims)  # init reward tensor
    # go through all pairs of histories
    for i, hist in enumerate(SAHists):
        for j, hist_ in enumerate(SAHists):
            hix, ix = _transition_ix(env, h, i, hist, j, hist_)
            hix = tuple([slice(env.N)]+list(hix))
            ix = tuple([slice(env.N)]+list(ix))
            Rh[hix] = env.R[ix]
    
    return Rh

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =        
#   Only histories of states
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
def StateHistsIx(env, h):
    '''
    Create possible state histories of length h 
    represented in indicies
    '''
    Hists = list(it.product(range(env.Z), repeat=h))

    # Remove squences that are not possible
    Tpossibles = env.T.sum(axis=tuple(range(env.N+2)[1:-1]))
    ZeroPossability = np.where(Tpossibles==0)
    NotPossibleTrans = list(zip(ZeroPossability[0],ZeroPossability[1]))

    def HistContainsNotPossibleTrans(hist, trans):
        contains = False
        for i in range(len(hist)-1):
            contains = hist[i:i+2] == trans
            if contains:
                break
        return contains

    PossibleHists = Hists.copy()
    for hist in Hists:
        for trans in NotPossibleTrans:
            if HistContainsNotPossibleTrans(hist, trans):
                PossibleHists.remove(hist)
                break
    return PossibleHists

def history_TransitionTensor(env, h):
    """
    Returns Transition Tensor of `env` with state history `h`[int]
    """
    Hists = StateHistsIx(env, h)

    Zh = len(Hists)
    Th_dims = list(env.T.shape)
    Th_dims[0] = Zh
    Th_dims[-1] = Zh
    Th = np.ones(Th_dims)

    for i, hist in enumerate(Hists):
        for j, hist_ in enumerate(Hists):
            possible = hist[1:] == hist_[:-1]  # Is the transition possible?
            Th[i, ..., j] = possible*env.T[hist[-1],...,hist_[-1]]

    return Th

def ObsHistsIx(env, h):
    '''
    Create possible observation histories of length `h`[int] 
    represented in indicies
    '''
    StateHists = StateHistsIx(env, h)
    ObsHists = list(it.product(range(env.Q), repeat=h))

    # Remove squences that are not possible to observe
    # for all agents
    # check all ohist elements
    PossibleObsHists = ObsHists.copy()
    for ohist in ObsHists:
        # whether they are observable by agent i
        observable = np.zeros(env.N)
        # go through all shist elements
        for shist in StateHists:
            # and check whether ohist can be observed from shist
            observable += np.prod([env.O[:, shist[k], ohist[k]]
                                   for k in range(len(shist))], axis=0)
        # if ohist can't be observed by any agent
        if np.allclose(observable, 0.0):
            # remove ohist from ObsHists
            PossibleObsHists.remove(ohist)

    return PossibleObsHists

def history_ObservationTensor(env, h):
    """
    Returns Observation Tensor of `env` with state history `h`[int]
    """
    StateHists = StateHistsIx(env, h)
    ObsHists = ObsHistsIx(env, h)

    Qh = len(ObsHists)
    Zh = len(StateHists)
    Oh = np.ones((env.N, Zh, Qh))

    for i, shist in enumerate(StateHists):
        for j, ohist in enumerate(ObsHists):
            Oh[:, i, j] = np.prod([env.O[:, shist[k], ohist[k]]
                                   for k in range(len(shist))], axis=0)
    
    return Oh

def history_RewardTensor(env, h):
    """
    Returns Reward Tensor of `env` with state history `h`[int]
    """
    StateHists = StateHistsIx(env, h)
    Zh = len(StateHists)
    dims = list(env.R.shape)
    dims[1] = Zh
    dims[-1] = Zh

    Rh = np.zeros(dims)
    for h, hist in enumerate(StateHists):
        for h_, hist_ in enumerate(StateHists):
            Rh[:, h, ..., h_] = env.R[:, hist[-1], ..., hist_[-1]]
    return Rh