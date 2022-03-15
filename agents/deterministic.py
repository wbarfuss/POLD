"""
Deterministic reinforcement Q-learning in the infitite memory-batch limit
"""
import numpy as np
import itertools as it
import scipy.linalg as la

class detMAE(object):
    """
    This is detMAE.py, a class for obtaining various bits for the deterministic
    limit of temporal difference reinforcement learning.
    """
    def __init__(self,
                 TransitionTensor,
                 RewardTensor,
                 ObservationTensor,
                 alpha,
                 beta,
                 gamma,
                 opteinsam=True,
                 roundingprec=9):
        """doc."""
        assert len(np.unique(TransitionTensor.shape[1:-1])) == 1,\
            "Transition tensor has different action sets sizes"
        assert len(np.unique(RewardTensor.shape[2:-1])) == 1,\
            "Reward tensor has different action sets sizes"
        assert len(ObservationTensor.shape) == 3,\
            "Observation tensor must be of shape O[N, Z, Q]"

        self.R = RewardTensor
        self.T = TransitionTensor
        self.O = ObservationTensor
        
        self.N = self.R.shape[0]  # the number of agents
        self.Z = self.T.shape[0]  # the number of states
        self.M = self.T.shape[1]  # the number of actions for each agent
        self.Q = self.O.shape[-1] # the number of observations

        # the agent's learning rate
        self.alpha = alpha  
        
        # the agent's exploitation level
        if hasattr(beta, '__iter__'):
            self.beta = np.array(beta)
        else:
            self.beta = np.repeat(beta, self.N)

        # the agent's discout factor
        if hasattr(gamma, '__iter__'):
            self.gamma = np.array(gamma)
        else:
            self.gamma = np.repeat(gamma, self.N)

        # other stuff
        self.Omega = self._obtain_OtherAgentsActionsSummationTensor()
        self.roundingprec = roundingprec
        self.opti = opteinsam  # optimize einsums

    # =========================================================================
    #   Behavior profiles
    # =========================================================================

    def zeroIntelligence_behavior(self):
        """Behavior profile with equal probabilities."""
        return np.ones((self.N, self.Q, self.M)) / float(self.M)

    def random_behavior(self, method="norm"):
        """Behavior profile with random probabilities."""
        if method=="norm":
            X = np.random.rand(self.N, self.Q, self.M)
            X = X / X.sum(axis=2).repeat(self.M).reshape(self.N, self.Q,
                                                         self.M)
        elif method == "diff":
            X = np.random.rand(self.N, self.Q, self.M-1)
            X = np.concatenate((np.zeros((self.N, self.Q, 1)),
                                np.sort(X, axis=-1),
                                np.ones((self.N, self.Q, 1))), axis=-1)
            X = X[:, :, 1:] - X[:, :, :-1]
        return X

    # =========================================================================
    #   Behavior profile averages
    # =========================================================================
    def obtain_Bios(self, X):
        """
        Belief that world is in stats s given i observe o under joint a
        (Bayes Rule)
        """
        i,s,o = 0, 1, 2
        pS = self.obtain_statedist(X)

        b = np.einsum(self.O, [i,s,o], pS, [s], [i,s,o], optimize=self.opti)
        bsum = b.sum(axis=1, keepdims=True)
        bsum = bsum + (bsum == 0)  # to avoid dividing by zero
        Biso = b /bsum
        Bios = np.swapaxes(Biso, 1,-1)
        
        return Bios

    def obtain_Xisa(self, X):
        i = 0
        a = 1
        s = 2
        o = 4
        
        args = [self.O, [i, s, o],
                X, [i, o, a], [i, s, a]]

        xisa = np.einsum(*args, optimize=self.opti)

        if not np.allclose(xisa.sum(-1), 1.0):
            print("[detMAE] Xisa no prob. Normalizing.", end="\r")
            xisa = xisa / xisa.sum(-1, keepdims=True)

        return xisa

    def obtain_Tioo(self, X, Bios=None, Xisa=None):
        Bios = self.obtain_Bios(X) if Bios is None else Bios
        Xisa = self.obtain_Xisa(X) if Xisa is None else Xisa
        
        i = 0  # agent i
        s = 1  # state s
        sprim = 2  # next state s'
        o = 3  # observation o
        oprim = 4 # next obs o'
        b2d = list(range(5, 5+self.N))  # all actions

        Y4einsum = list(it.chain(*zip(Xisa,
                                      [[s, b2d[a]] for a in range(self.N)])))

        args = [Bios, [i, o, s]] + Y4einsum + [self.T, [s]+b2d+[sprim],
                self.O, [i, sprim, oprim], [i, o, oprim]]
        return np.einsum(*args, optimize=self.opti)

    def obtain_Tioao(self, X, Bios=None, Xisa=None):
        """
        Effective environmental transition model for agent i
        """
        Bios = self.obtain_Bios(X) if Bios is None else Bios
        Xisa = self.obtain_Xisa(X) if Xisa is None else Xisa
        
        i = 0  # agent i
        a = 1  # its action a
        s = 2  # the current state
        sprim = 3  # the next state
        o = 4
        oprim = 5
        j2k = list(range(6, 6+self.N-1))  # other agents
        b2d = list(range(6+self.N-1, 6+self.N-1 + self.N))  # all actions
        e2f = list(range(5+2*self.N, 5+2*self.N + self.N-1))  # all other acts

        sumsis = [[j2k[l], s, e2f[l]] for l in range(self.N-1)]  # sum inds
        otherY = list(it.chain(*zip((self.N-1)*[Xisa], sumsis)))

        args = [self.Omega, [i]+j2k+[a]+b2d+e2f,
                Bios, [i, o, s]] + otherY + [self.T, [s]+b2d+[sprim],
                self.O, [i, sprim, oprim], [i, o, a, oprim]]
        return np.einsum(*args, optimize=self.opti)

    def obtain_Rioa(self, X, Bios=None, Xisa=None):
        Bios = self.obtain_Bios(X) if Bios is None else Bios
        Xisa = self.obtain_Xisa(X) if Xisa is None else Xisa
        
        i = 0
        a = 1
        s = 2
        sprim = 3
        o = 4
        j2k = list(range(5, 5+self.N-1))  # other agents
        b2d = list(range(5+self.N-1, 5+self.N-1 + self.N))  # all actions
        e2f = list(range(4+2*self.N, 4+2*self.N + self.N-1))  # all other acts
 
        sumsis = [[j2k[l], s, e2f[l]] for l in range(self.N-1)]  # sum inds
        otherY = list(it.chain(*zip((self.N-1)*[Xisa], sumsis)))

        args = [self.Omega, [i]+j2k+[a]+b2d+e2f, Bios, [i, o, s]] +\
                otherY + [self.T, [s]+b2d+[sprim], self.R, [i, s]+b2d+[sprim],
                [i, o, a]]

        return np.einsum(*args, optimize=self.opti)

    def obtain_Rio(self, X, Bios=None, Xisa=None):
        """
        Reward of agent i as observed in o, given all agents behave according
        to X.
        """
        Bios = self.obtain_Bios(X) if Bios is None else Bios
        Xisa = self.obtain_Xisa(X) if Xisa is None else Xisa
        
        i = 0  # agent i
        s = 1  # state s
        sprim = 2  # next state s'
        o = 3  # observation o
        b2d = list(range(4, 4+self.N))  # all actions
    
        Y4einsum = list(it.chain(*zip(Xisa,
                                 [[s, b2d[a]] for a in range(self.N)])))
        
        args = [Bios, [i, o, s]] + Y4einsum + [self.T, [s]+b2d+[sprim],
                self.R, [i, s]+b2d+[sprim], [i, o]]
        return np.einsum(*args, optimize=self.opti)

    def obtain_Vio(self, X, Rio=None, Tioo=None, Bios=None, Xisa=None):
        Bios = self.obtain_Bios(X) if Bios is None else Bios
        Xisa = self.obtain_Xisa(X) if Xisa is None else Xisa
        Rio = self.obtain_Rio(X, Bios=Bios, Xisa=Xisa) if Rio is None else Rio
        Tioo = self.obtain_Tioo(X, Bios=Bios, Xisa=Xisa) if Tioo is None\
            else Tioo
        
        i = 0
        o = 1
        op = 2

        n = np.newaxis
        Mioo = np.eye(self.Q)[n,:,:] - self.gamma[:, n, n] * Tioo
        invMioo = self._vect_matrix_inverse(Mioo)

        return (1-self.gamma[:, n]) * np.einsum(invMioo, [i, o, op],
                                                Rio, [i, op],
                                                [i, o], optimize=self.opti)
    
    def obtain_Qioa(self, X, Rioa=None, Vio=None, Tioao=None,
                    Bios=None, Xisa=None):
    
        Vio = self.obtain_Vio(X, Bios=Bios, Xisa=Xisa) if Vio is None else Vio        
        Rioa = self.obtain_Rioa(X, Bios=Bios, Xisa=Xisa) if Rioa is None\
            else Rioa
        Tioao = self.obtain_Tioao(X, Bios=Bios, Xisa=Xisa) if Tioao is None\
            else Tioao

        nextQioa = np.einsum(Tioao, [0,1,2,3], Vio, [0,3], [0,1,2],
                             optimize=self.opti)

        n = np.newaxis
        return (1-self.gamma[:,n,n]) * Rioa + self.gamma[:,n,n]*nextQioa

    # def obtain_Vio_alt(self, X):
    #     i = 0
    #     o = 1
    #     s = 2
        
    #     Vis = self.obtain_Vis(X)
    #     Bios = self.obtain_Bios(X)
        
    #     return np.einsum(Bios, [i, o, s], Vis, [i, s], [i, o], 
    #                      optimize=self.opti)
    
    # def obtain_Qioa_alt(self, X):
    #     Rioa = self.obtain_Rioa(X)
    #     Vio = self.obtain_Vio_alt(X)
    #     Tioao = self.obtain_Tioao(X)

    #     nextQioa = np.einsum(Tioao, [0,1,2,3], Vio, [0,3], [0,1,2],
    #                          optimize=self.opti)

    #     n = np.newaxis
    #     return (1-self.gamma[:,n,n]) * Rioa + self.gamma[:,n,n]*nextQioa
    # =========================================================================
    # needs for state distribution
    def obtain_Tss(self, X):
        """Effective Markov Chain transition tensor."""
        Xisa = self.obtain_Xisa(X)
        x4einsum = list(it.chain(*zip(Xisa,
                                      [[0, i+1] for i in range(self.N)])))
        x4einsum.append(np.arange(self.N+1).tolist())  # output format
        Xs = np.einsum(*x4einsum, optimize=self.opti)

        return np.einsum(self.T, np.arange(self.N+2).tolist(),  # trans. tensor
                         Xs, np.arange(self.N+1).tolist(),      # policies
                         [0, self.N+1], optimize=self.opti)     # output format

    def obtain_Tisas(self, X):
        """
        Effective environmental transition model for agent i
        """
        Xisa = self.obtain_Xisa(X)

        i = 0  # agent i
        a = 1  # its action a
        s = 2  # the current state
        sprim = 3  # the next state
        j2k = list(range(4, 4+self.N-1))  # other agents
        b2d = list(range(4+self.N-1, 4+self.N-1 + self.N))  # all actions
        e2f = list(range(3+2*self.N, 3+2*self.N + self.N-1))  # all other acts

        # get arguments ready for function call
        # # 1# other policy X
        sumsis = [[j2k[o], s, e2f[o]] for o in range(self.N-1)]  # sum inds
        otherX = list(it.chain(*zip((self.N-1)*[Xisa], sumsis)))

        args = [self.Omega, [i]+j2k+[a]+b2d+e2f,
                self.T, [s]+b2d+[sprim]] + otherX + [[i, s, a, sprim]]

        return np.einsum(*args, optimize=self.opti)

    def obtain_Risa(self, X):
        """
        Reward of agent i in state s under action a,
        given other agents behavior according to X.
        """
        Xisa = self.obtain_Xisa(X)
        i = 0  # agent i
        a = 1  # its action a
        s = 2  # the current state
        sprim = 3  # the next state
        j2k = list(range(4, 4+self.N-1))  # other agents
        b2d = list(range(4+self.N-1, 4+self.N-1 + self.N))  # all actions
        e2f = list(range(3+2*self.N, 3+2*self.N + self.N-1))  # all other acts

        # get arguments ready for function call
        # # 1# other policy X
        sumsis = [[j2k[l], l, e2f[l]] for l in range(self.N-1)]  # sum inds
        otherX = list(it.chain(*zip((self.N-1)*[Xisa], sumsis)))

        args = [self.Omega, [i]+j2k+[a]+b2d+e2f,
                self.R, [i, s]+b2d+[sprim],
                self.T, [s]+b2d+[sprim]] + otherX + [[i, s, a]]

        return np.einsum(*args, optimize=self.opti)

    def obtain_Ris(self, X):
        """
        Reward of agent i in state s, given all agents behavior according to X.
        """
        Xisa = self.obtain_Xisa(X)
        i = 0  # agent i
        s = 1  # state s
        sprim = 2  # next state s'
        b2d = list(range(3, 3+self.N))  # all actions

        x4einsum = list(it.chain(*zip(Xisa,
                                      [[s, b2d[a]] for a in range(self.N)])))

        # einsum argument
        args = [self.R, [i, s]+b2d+[sprim],
                self.T, [s]+b2d+[sprim]] +\
             x4einsum +\
             [[i, s]]

        return np.einsum(*args, optimize=self.opti)

    def obtain_Vis(self, X):
        """State Value of state s for agent i"""
        i = 0
        s = 1
        sp = 2

        Ris = self.obtain_Ris(X)
        Tss = self.obtain_Tss(X)
        # new
        n = np.newaxis
        Miss = np.eye(self.Z)[n,:,:] - self.gamma[:, n,n] * Tss[n,:,:]
        invMiss = self._vect_matrix_inverse(Miss)

        return (1-self.gamma[:, n]) * np.einsum(invMiss, [i, s, sp],
                                                Ris, [i, sp],
                                                [i, s], optimize=self.opti)
    def obtain_Qisa(self, X):
        """Current state action value
        Q = (1-gamma)*Risa + gamma * VXT
        """
        Risa = self.obtain_Risa(X)
        Vis = self.obtain_Vis(X)
        Tisas = self.obtain_Tisas(X)

        nextQisa = np.einsum(Tisas, [0,1,2,3], Vis, [0,3], [0,1,2],
                             optimize=self.opti)
        
        n = np.newaxis
        return (1-self.gamma[:,n,n]) * Risa + self.gamma[:,n,n]*nextQisa

    # =========================================================================
    #   HELPER
    # =========================================================================
    @staticmethod
    def _vect_matrix_inverse(A):
        """
        Vectorized matrix inverse.

        first dimension of A is running index, the last two dimensions are
        the matrix dimensions
        """
        identity = np.identity(A.shape[2], dtype=A.dtype)
        return np.array([np.linalg.solve(x, identity) for x in A])

    def _obtain_OtherAgentsActionsSummationTensor(self):
        """For use in Einstein Summation Convention.

        To sum over the other agents and their respective actions.
        """
        dim = np.concatenate(([self.N],  # agent i
                              [self.N for _ in range(self.N-1)],  # other agnt
                              [self.M],  # agent a of agent i
                              [self.M for _ in range(self.N)],  # all acts
                              [self.M for _ in range(self.N-1)]))  # other a's
        Omega = np.zeros(dim.astype(int), int)

        for index, _ in np.ndenumerate(Omega):
            I = index[0]
            notI = index[1:self.N]
            A = index[self.N]
            allA = index[self.N+1:2*self.N+1]
            notA = index[2*self.N+1:]

            if len(np.unique(np.concatenate(([I], notI)))) is self.N:
                # all agents indicides are different

                if A == allA[I]:
                    # action of agent i equals some other action
                    cd = allA[:I] + allA[I+1:]  # other actionss
                    areequal = [cd[k] == notA[k] for k in range(self.N-1)]
                    if np.all(areequal):
                        Omega[index] = 1

        return Omega

    def _obtain_statdist(self, Tss, tol=1e-10, adapt_tol=True,
                        verbose=True):
        """Obtain stationary distribution for X"""

        oeival, oeivec = la.eig(Tss, right=False, left=True)
        oeival = oeival.real
        oeivec = oeivec.real
        mask = abs(oeival - 1) < tol
        
        
        printflag = False
        
        
        if adapt_tol and np.sum(mask) != 1:
            # not ONE eigenvector found AND tolerance adaptation true
            sign = 2*(np.sum(mask)<1) - 1 # 1 if sum(mask)<1, -1 if sum(mask)>1

            while np.sum(mask) != 1 and tol < 1.0 and tol > 1e-24:
                tol = tol * 10**(int(sign))
                if verbose:
                    print(f"[MAE-statdist] Adapting tolerance to {tol}")
                    
                printflag = True
                
                
                mask = abs(oeival - 1) < tol

        meivec = oeivec[:, mask]
        dist = meivec / meivec.sum(axis=0, keepdims=True)
        dist[dist < tol] = 0
        dist = dist / dist.sum(axis=0, keepdims=True)
        
        if printflag:
            print(dist)
            print(sign)
        
        # if verbose:
        #     if len(dist[0]) > 1:
        #         print("more than 1 eigenvector found")
        #         print("taking average")
        #         dist = dist.mean(1, keepdims=True)
        #     if len(dist[0]) < 1:
        #         print("less than 1 eigenvector found")

        return dist

    def obtain_obsdist(self, X):
        Tioo = self.obtain_Tioo(X)

        Dis = np.zeros((self.N, self.Q))
        for i in range(self.N):
            pO = self._obtain_statdist(Tioo[i])
            if len(pO[0]) > 1:
                print("more than 1 obs-eigenvector found")
                # print(Tioo.round(2))
                print(pO.round(2))
                
                print("taking average")
                pO = pO.mean(1, keepdims=True)
                
                # print("taking first one")
                # pO = pO[:, 0]
                
                # print(pO.round(2))
                # print()
            Dis[i] = pO.flatten()
        return Dis

    def obtain_statedist(self, X):
        Tss = self.obtain_Tss(X)

        pS = self._obtain_statdist(Tss)
        if len(pS[0]) > 1:
                print("more than 1 state-eigenvector found")
                nr = len(pS[0])
                choice = np.random.randint(nr)
                
                print("taking random one: ", choice)
                pS = pS[:, choice]
                
        return pS.flatten()

    def TDstep(self, X):
        TDe = self.TDerror(X)

        n = np.newaxis

        num = X * np.exp(self.alpha * self.beta[:,n,n] * TDe)
        num = np.round(num, self.roundingprec)

        den = np.reshape(np.repeat(np.sum(num, axis=2), self.M),
                         X.shape)

        return num / den

class detQ(detMAE):

    def __init__(self,
                 TranstionTensor,
                 RewardTensor,
                 ObservationTensor,
                 alpha,
                 beta,
                 gamma,
                 opteinsum=True,
                 FinalStates=None,
                 roundingprec=9):

        detMAE.__init__(self,
                        TranstionTensor,
                        RewardTensor,
                        ObservationTensor,
                        alpha,
                        beta,
                        gamma,
                        opteinsum,
                        roundingprec)
        if FinalStates is not None:
            FS = np.array(FinalStates, dtype=int)
        else:
            FS = np.zeros(self.Z, dtype=int)
        #                                i  s  o
        self.FinObs = np.einsum(self.O, [1, 2, 0], FS, [2], [1, 0],
                                optimize=self.opti)

    # =========================================================================
    #   Temporal difference error
    # =========================================================================

    def TDerror(self, X, norm=True):
        Xisa = self.obtain_Xisa(X)
        Bios = self.obtain_Bios(X)
        
        R = self.obtain_Rioa(X, Bios=Bios, Xisa=Xisa)
        MaxQ = self.obtain_MaxQioa(X, Rioa=R, Bios=Bios, Xisa=Xisa)
        
        n = np.newaxis

        # TDe = (1-self.gamma[:,n,n])*R\
        # + self.gamma[:,n,n]*MaxQ\
        # - 1/self.beta * np.ma.log(X)
        
        TDe = (1-self.gamma[:,n,n])*R\
            + (1-self.FinObs[:,:,n])*self.gamma[:,n,n]*MaxQ\
            - 1/self.beta * np.ma.log(X)

        if norm:
            TDe = TDe - TDe.mean(axis=2, keepdims=True)
        TDe = TDe.filled(0)
        return TDe

    # =========================================================================
    #   Behavior profile averages
    # =========================================================================

    def obtain_MaxQioa(self, X, Qioa=None, Rioa=None, Vio=None, Tioao=None, 
                       Bios=None, Xisa=None):      
        Bios = self.obtain_Bios(X) if Bios is None else Bios
        Xisa = self.obtain_Xisa(X) if Xisa is None else Xisa
        Qioa = self.obtain_Qioa(X, Rioa=Rioa, Vio=Vio, Tioao=Tioao,
                                Bios=Bios, Xisa=Xisa) if Qioa is None else Qioa
  
        i = 0  # agent i
        a = 1  # its action a
        s = 2  # the current state
        sprim = 3  # the next state
        o = 4  # observatio o
        oprim = 5  # next observatio 
        j2k = list(range(6, 6+self.N-1))  # other agents
        b2d = list(range(6+self.N-1, 6+self.N-1 + self.N))  # all actions
        e2f = list(range(5+2*self.N, 5+2*self.N + self.N-1))  # all other acts

        sumsis = [[j2k[l], s, e2f[l]] for l in range(self.N-1)]  # sum inds
        otherY = list(it.chain(*zip((self.N-1)*[Xisa], sumsis)))
            
        args = [self.Omega, [i]+j2k+[a]+b2d+e2f, Bios, [i, o, s]] + otherY +\
               [self.T, [s]+b2d+[sprim], self.O, [i, sprim, oprim], 
                Qioa.max(axis=-1), [i, oprim], [i, o, a]]
        return np.einsum(*args, optimize=self.opti)
