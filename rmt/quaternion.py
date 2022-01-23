import numpy as np

e0 = np.eye(2)
e1 = np.array([[1j,0], [0, -1j]])
e2 = np.array([[0, 1], [-1, 0]])
e3 = np.array([[0, 1j], [1j, 0]])


def quaternion(a, b, c, d):
    return np.kron(a, e0)+np.kron(b, e1)+np.kron(c, e2)+np.kron(d, e3)


def symplectic(a, b, c, d):
    q = quaternion(a, b, c, d)
    return 0.5*(q+q.T.conj)


def symplectic_form(n):
    return -np.diag(np.ones(n-1), -1)+np.diag(np.ones(n-1), 1)