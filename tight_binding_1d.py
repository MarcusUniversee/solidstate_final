from scipy.sparse import diags, lil_matrix
from scipy.linalg import ishermitian
import numpy as np
import matplotlib.pyplot as plt

N = 9
t = 1

# TASK 1

#tridiagonal
ones = np.ones(N-1)
H_P = diags([-t*ones, -t*ones], [1, -1], shape=(N, N), format='lil')

#corner terms for periodic conditions
H_P[0, N-1] = -t
H_P[N-1, 0] = -t

H_P = H_P.toarray()
print("==== periodic boundary conditions: ====")
print(H_P)
print("is hermitian: ", ishermitian(H_P))

H_O = diags([-t*ones, -t*ones], [1, -1], shape=(N, N), format='lil')

H_O = H_O.toarray()
print("\n")
print("====== open boundary conditions: ======")
print(H_O)
print("is hermitian: ", ishermitian(H_O))


# TASK 2
print("\n\n")
eigenvalues_p, eigenvectors_p = np.linalg.eigh(H_P)
eigenvalues_o, eigenvectors_o = np.linalg.eigh(H_O)

print("eigenvalues for periodic:")
print(eigenvalues_p)
print("\n")
print("eigenvalues for open:")
print(eigenvalues_o)