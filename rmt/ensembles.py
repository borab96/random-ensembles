
import scipy.stats as scs
import numpy as np
from .quaternion import symplectic


def ge(n):
    """
    :param n: size of random matrix
    :return: random n by n matrix, drawn from the standard Gaussian ensemble
    """
    return scs.norm().rvs(size=[int(n), int(n)])


def goe(n):
    """
    :param n: size of random matrix
    :return: random n by n matrix, drawn from the standard Gaussian orthogonal ensemble
    """
    return 0.5*(ge(n)+ge(n).T)


def gse(n):
    """
    :param n: size of random matrix
    :return: random n by n matrix, drawn from the standard Gaussian symplectic ensemble
    """
    return symplectic(ge(n), ge(n), ge(n), ge(n))


def gue(n):
    """
    :param n: size of random matrix
    :return: random n by n matrix, drawn from the standard Gaussian unitary ensemble
    """
    ge_C = ge(n) + 1j * ge(n)
    return 0.5*(ge_C+ge_C.T.conj())


def ginibre(n, complex=True):
    """
    This is a common non-Hermitian ensemble

    :param n: size of random matrix
    :param complex: if true, complex Ginibre ensemble, otherwise real
    :return: random n by n matrix, drawn from the standard Gaussian ginibre ensemble
    """
    if complex:
        return ge(n) * np.sqrt(1 / (2 * n)) + 1j * ge(n) * np.sqrt(1 / (2 * n))
    else:
        return  ge(n) * np.sqrt(1 /n)



def le(n, alpha, beta=0):
    """
    Draw from the Levy-stable ensemble

    :param n: size of random matrix
    :param alpha: parameter controlling the asymptotic power law of the distribution tails, between 0 and 2
    :param beta: skew of the distribution, between -1 and 1
    :return: random n by n matrix, drawn from the Levy ensemble
    """
    rv = scs.levy_stable(alpha, beta)
    return rv.rvs(size=[int(n), int(n)])


def disorder(random_matrix, rv, seed=None):
    """
    Adds (quenched) disorder drawn from a positive distribution to the random matrix.
    See PHYSICAL REVIEW E 77, 011122  (2008)

    :param random_matrix: random matrix drawn from any ensemble
    :param rv: positive disorder random variable
    :param seed: setting this to not None quenches the disorder
    :return: disordered random matrix
    """
    return random_matrix/np.sqrt(rv.rvs(random_state=seed)/rv.mean())