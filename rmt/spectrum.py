import numpy as np
from scipy.special import gamma


def mean_norm(eig, n, beta):
    """
    normalization/unfolding procedure for random spectrum
    :param eig: spectrum of random matrix
    :param n: size of random matrix
    :param beta: dyson index [1, 2, 4]
    :return: normalized random spectrum
    """
    f = lambda x: n/2+1/(np.pi*beta)*(n*beta*np.arcsin(np.sqrt(1/(n*2*beta))*x)+0.5*x*np.sqrt(2*n*beta-x**2))
    return np.piecewise(eig, [eig <= -np.sqrt(2*beta*n), eig >= np.sqrt(2*beta*n)], [0, n, f])


def level_space(generator, *args, n_mc=200, hermitian=True, normalize_mean=True, tol=0.001):
    """
    :param generator: the generating function of the desired ensemble, gue, gse etc.
    :param args: arguments of the generating function
    :param n_mc: number of Monte Carlo steps to sample the distribution
    :param hermitian: If True, use hermitian linalg methods
    :param normalize_mean: If True, mean spacing is normalized to 1 using method `mean_norm`
    :param tol: tolerance parameter for numerical stability. A small strictly positive number
    :return: numpy array of level spacing distribution samples.
    """
    spacings = []
    if "gse" in generator.__name__.lower():
        dyson_index = 4
    elif "goe" in generator.__name__.lower():
        dyson_index = 1
    elif "gue" in generator.__name__.lower():
        dyson_index = 2
    else:
        dyson_index = 0
    if "ginibre" in generator.__name__.lower():
        hermitian = False
    for _ in range(n_mc):
        rm = generator(*args)
        if hermitian:
            e, _ = np.linalg.eigh(rm)
        else:
            e, _ = np.linalg.eigh(rm)
        if normalize_mean and bool(dyson_index):
            e = mean_norm(e, args[0], dyson_index)
        eig_spaces = [e[i+1] - e[i] for i in range(args[0]-1) if e[i+1]-e[i] > tol]
        spacings.extend(eig_spaces)
    return np.array(spacings)


def spectral_density(rm, hermitian=True):
    if hermitian:
        e = np.linalg.eigvalsh(rm)
    else:
        e = np.linalg.eigvals(rm)
    return e/np.sqrt(len(e))


def wigner_surmise(s, beta):
    """
    Theoretical distribution of level spacings. This should match the level spacing distribution of the GOE, GUE and GSE
    :param s: linspaced axis vector
    :param beta: dyson index [1, 2, 4]
    :return: wigner surmise for given dyson index
    """
    a = 2*(gamma((beta+2)/2))**(beta+1)/(gamma((beta+1)/2))**(beta+2)
    b = (gamma((beta+2)/2))**2/(gamma((beta+1)/2))**2
    return a * (s**beta) * np.exp(-b*s**2)