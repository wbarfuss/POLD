'''
Update / rewrite of quiverplot
a new beginning . rock and roll.
'''
import numpy as np
import matplotlib.pyplot as plt
import itertools as it

# from pyDOE import lhs
from scipy.stats import qmc


#%%
# Note: Could be sample via latin hyper cube sampling (to be explored)
def _policies(dims, xinds, yinds, xval, yval, NrRandom):
    """
    Creates policies for one ax plot point.
    """
    N, C, M = dims
    # Xs = np.random.rand(NrRandom, N, C, M)  # random policies
    # Xs = lhs(N*C*M, NrRandom).reshape(NrRandom, N, C, M)
    sampler = qmc.LatinHypercube(d=N*C*M)
    Xs = sampler.random(n=NrRandom).reshape(NrRandom, N, C, M)
    Xs = Xs / Xs.sum(axis=-1, keepdims=True)  # properly normalised
    
    xi, xc, xa = xinds; xa_ = tuple(set(range(M)) - set([xa]))
    yi, yc, ya = yinds; ya_ = tuple(set(range(M)) - set([ya]))
    
    Xs[:, xi, xc, xa] = xval  # setting x and y values
    Xs[:, yi, yc, ya] = yval
    
    Xs[:,xi,xc,xa_] = (1-Xs[0, xi, xc, xa]) * Xs[:,xi,xc,xa_]\
        / np.sum(Xs[:,xi,xc,xa_], axis=-1, keepdims=True)
    Xs[:,yi,yc,ya_] = (1-Xs[0, yi, yc, ya]) * Xs[:,yi,yc,ya_]\
        / np.sum(Xs[:,yi,yc,ya_], axis=-1, keepdims=True)
    
    return Xs


#%%

def _iterate_policies(Xs, agents, steps):
    """Iterate policies forward"""
    if steps >= 1:
        newXs = []
        for X in Xs:
            newX = X
            for t in range(steps):
                newX = agents.TDstep(newX)
            newXs.append(newX)
        return np.array(newXs)
    else:
        return Xs    

def TDerror_difference(Xs, agents):
    """Compute TDerros for Xs."""
    return np.array([agents.TDerror(X) for X in Xs])

def DeltaX_difference(Xs, agents):
    """Compute X(t-1)-X(t) for Xs."""
    return np.array([agents.TDstep(X) - X for X in Xs])

def _getNCM(agents):
    M = agents.R.shape[2]  # Number of actions
    N = agents.R.shape[0]  # Number of agents
    C = agents.Q if hasattr(agents, "Q") else agents.Z  # Number of conditions
    return N, C, M

# Note: we could also generate the data with an unregular meshgrid
#       e.g., from a latin grid ( to be explored )
def _data_to_plot(agents, pAs, xinds, yinds, NrRandom, difffunc, 
                  policies_iter_steps=0, verbose=False):
    N, C, M = _getNCM(agents)
    l = len(pAs)
    X, Y = np.meshgrid(pAs, pAs)
    dX, dY = np.zeros((l, l, NrRandom)), np.zeros((l, l, NrRandom))

    for i, pX in enumerate(pAs):
        for j, pY in enumerate(pAs):
            if verbose:
                print("[plot] generating data",
                      str(np.round((i*l+j)/l**2,2)*100)[:2], "%")
            PIs = _policies(dims=(N,C,M), xinds=xinds, yinds=yinds,
                            xval=pX, yval=pY, NrRandom=NrRandom)
            PIs = _iterate_policies(PIs, agents, steps=policies_iter_steps)
            dPIs = difffunc(PIs, agents)
            dX[j, i] = np.moveaxis(dPIs, 0, -1)[xinds]
            dY[j, i] = np.moveaxis(dPIs, 0, -1)[yinds]
            
    return X, Y, dX, dY


#%%
def _plot(dX, dY, X, Y, ax=None, sf=1.0, col='LEN', cmap="viridis",
          kind='quiver+samples', lw=1.0, dens=0.75):
    """
    Plots the quivers for one condition into one axes
    """
    if ax is None:
        _, ax = plt.subplots(1,1, figsize=(4,4))
        
    if kind == "streamplot":
        DX = dX.mean(axis=-1)
        DY = dY.mean(axis=-1)
        LEN = (DX**2 + DY**2)**0.5
        col = LEN if col=="LEN" else col
        ax.streamplot(X, Y, DX, DY, color=LEN, linewidth=lw, cmap=cmap,
                      density=dens)
        
    elif kind.startswith('quiver'):
        # quiver keywords
        qkwargs = {"units":"xy", "angles": "xy", "scale":None,
                   "scale_units": "xy", "headwidth": 5, "pivot": "tail"}
    
        if kind.endswith('samples'):  # plot inidividual samples
            Nr = dX.shape[-1]
            for i in range(Nr):
                DX = dX[:, :, i]
                DY = dY[:, :, i]
                
                if col == "LEN":
                    LEN = (DX**2 + DY**2)**0.5 
                    ax.quiver(X, Y, *_scale(DX, DY, sf), LEN, alpha=1/Nr,
                              cmap=cmap, **qkwargs)
                else:
                    ax.quiver(X, Y, *_scale(DX, DY, sf), alpha=1/Nr, color=col,
                              **qkwargs) 

        DX = dX.mean(axis=-1)
        DY = dY.mean(axis=-1)
        if col == "LEN":
            LEN = (DX**2 + DY**2)**0.5
            ax.quiver(X, Y, *_scale(DX, DY, sf), LEN, cmap=cmap, **qkwargs)
        else:
            ax.quiver(X, Y, *_scale(DX, DY, sf), color=col, **qkwargs)        
    
    
    ax.set_xlim(-0.025, 1.025); ax.set_ylim(-0.025, 1.025)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    return ax

def _scale(x, y, a):
    """
    Scales length of x,y vec accoring to length ** a
    """
    l = (x**2 + y**2)**(1/2)
    l = l + (l==0)
    k = l**a
    return k/l * x, k/l * y


def plot_quiver(agents, x, y, pAs, NrRandom=3, axes=None, diff="TDe", sf=0.5,
                kind='quiver+samples', lw=1.0, dens=0.75, policies_iter_steps=0, 
                verbose=False, acts=None, conds=None, col='LEN', cmap='viridis'):
    """
    Create quiver plot
    """
    # Checks and balances
    xlens, amx, lens = _check_and_balances(x, y)
    
    # Fig and Axes
    axes = _prepare_axes(axes, xlens)

    # The Plots
    difffunc = TDerror_difference if diff=="TDe" else DeltaX_difference
    for i, (xinds, yinds) in enumerate(zip(it.product(*x), it.product(*y))):
        X, Y, dX, dY = _data_to_plot(agents, pAs, xinds, yinds, NrRandom,
                                     difffunc, verbose=verbose,
                                     policies_iter_steps=policies_iter_steps)
        axes[i] = _plot(dX, dY, X, Y, ax=axes[i], sf=sf, kind=kind,
                        lw=lw, dens=dens, col=col, cmap=cmap)
        
        
    # Decorations
    lens = max(xlens)
    if acts is None:
        acts = [f'act.{i}' for i in range(agents.M)]
    if conds is None:
        conds = [f'Cond. {i}' for i in range(lens)]
    
    if lens > 1 and amx == 1: # plotting over conditions
        for c, ax in enumerate(axes):
            ax.set_title(conds[c])
            ax.set_xlabel(f"Agnt {x[0][0]+1}'s prob. for {acts[x[2][0]]}")
            ax.set_ylabel(f"Agnt {y[0][0]+1}'s prob. for {acts[y[2][0]]}")
            ax.xaxis.labelpad = -8; ax.yaxis.labelpad = -8
    return axes


def plot_trajectories(Xtrajs, x, y, lss=["-"], lws=[3], cols=["r"],
                      alphas=[1.0], fprs=None,  mss=[None], msss=[0],
                      axes=None):
    # Checks and balances
    xlens, amx, lens = _check_and_balances(x, y)
    
    # Fig and Axes
    axes = _prepare_axes(axes, xlens)

    # Fixed point reached?
    if fprs is None:
        fprs = [False for _ in range(len(Xtrajs))]

    # The Plots
    cols = it.cycle(cols); lss = it.cycle(lss); lws = it.cycle(lws);
    mss = it.cycle(mss); msss = it.cycle(msss)
    alphas = it.cycle(alphas);
    for i, (xinds, yinds) in enumerate(zip(it.product(*x), it.product(*y))):
        for j, Xtraj in enumerate(Xtrajs):
            xs = np.moveaxis(Xtraj, 0, -1)[xinds]
            ys = np.moveaxis(Xtraj, 0, -1)[yinds]
            
            c =  next(cols)
            w = next(lws)
            m = next(mss)
            ms = next(msss)
            alph = next(alphas)
            axes[i].plot(xs, ys, lw=w, ls=next(lss), color=c, alpha=alph,
                         marker=m, markersize=ms)
            
            axes[i].scatter(xs[0], ys[0], color=c, marker='x', s=12*w, alpha=alph)
            if fprs[j]:
                axes[i].scatter(xs[-1], ys[-1], color=c, marker='o', s=20*w,
                                alpha=alph)
            
    return axes
            # beginning and end point

def _check_and_balances(x,y):
    xlens = np.array(list(map(len,x))); ylens = np.array(list(map(len,y)))
    assert sum(xlens == 1) >= 2, "min 2 elements must have length 1"
    assert sum(ylens == 1) >= 2, "min 2 elements must have length 1"
    amx = np.argmax(xlens); amy = np.argmax(ylens)
    if amx > 1:
        assert amx == amy, "x and y must have same iter"
        assert x[amx] == y[amy], "x and y must have same iter"  
    lens = max(xlens)
    return xlens, amx, lens          

def _prepare_axes(axes, xlens):      
    lens = max(xlens)
    if axes is None:
        fig, axes = plt.subplots(1, lens, figsize=(3*lens, 2.8))
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    assert len(axes) == lens, "Number of axes must fit to x and y"
    
    return axes
# %%
