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


def level_space(generator, *args, n_mc = 200, hermitian=True, normalize_mean="analytic", dyson_index_override=False):
    spacings = []
    if "gse" in generator.__name__.lower():
        dyson_index = 4
    elif "goe" in generator.__name__.lower():
        dyson_index = 1
    elif "gue" in generator.__name__.lower():
        dyson_index = 2
    else:
        dyson_index = 0
    if dyson_index_override:
        dyson_index = dyson_index_override

    for _ in tqdm(range(n_mc)):
        rm = generator(*args)
        if hermitian:
            e, _ = np.linalg.eigh(rm)
        else:
            e, _ = np.linalg.eigh(rm)
        if normalize_mean=="analytic" and bool(dyson_index):
            e = F(e, args[0], dyson_index)
            eig_spaces = [e[i+1] - e[i] for i in range(args[0]-1) if e[i+1]-e[i]>0.001]
        elif normalize_mean=="brute_force":
            kernel = sc.stats.gaussian_kde(e)
            t = np.linspace(min(e), max(e), len(e))
            def scipy_staircase(e):
                return sc.integrate.quad(lambda s: kernel.evaluate(s), -np.inf, e)[0]
            eig_spaces = [scipy_staircase(e[i+1])*len(e) - scipy_staircase(e[i])*len(e)
                          for i in range(args[0]-1) if e[i+1]-e[i]>0.001]
        else:
            eig_spaces = [e[i+1] - e[i] for i in range(args[0]-1) if e[i+1]-e[i]>0.001]
        spacings += eig_spaces
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
