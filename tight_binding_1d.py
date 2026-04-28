from scipy.sparse import diags, lil_matrix
from scipy.linalg import ishermitian
import numpy as np
import matplotlib.pyplot as plt

N = 100
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

fig, ax = plt.subplots(figsize=(8, 5))

ax.set_title("Energy spectrum periodic and open systems at N=100, t=1")
ax.scatter(range(len(eigenvalues_p)), eigenvalues_p, label='Periodic (PBC)', marker='o', zorder=3)
ax.scatter(range(len(eigenvalues_o)), eigenvalues_o, label='Open (OBC)', marker='D', facecolors='none', edgecolors='C1', zorder=3)
ax.set_xlabel('State index (sorted by energy)')
ax.set_ylabel('Energy E')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('energy_spectrum.png', dpi=150)
plt.show()

x = np.arange(N)
labels_p = ['Ground state', '1st excited', '2nd excited']

fig, ax = plt.subplots(figsize=(8, 5))
for i in range(3):
    psi_sq = np.abs(eigenvectors_p[:, i])**2
    ax.plot(x, psi_sq, marker='o', label=f'{labels_p[i]} (E={eigenvalues_p[i]:.3f})')
ax.axhline(1 / N, color='k', lw=1.2, ls='--', alpha=0.6, label=f'Uniform density $1/N = {1/N:.3f}$')
ax.set_xlabel('Site index x')
ax.set_ylabel(r'$|\psi_n(x)|^2$')
ax.set_title('Periodic (PBC) — Ground state and first two excited states at N=100, t=1')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('wavefunctions_periodic.png', dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
for i in range(3):
    psi_sq = np.abs(eigenvectors_o[:, i])**2
    ax.plot(x, psi_sq, marker='o', label=f'{labels_p[i]} (E={eigenvalues_o[i]:.3f})')
ax.set_xlabel('Site index x')
ax.set_ylabel(r'$|\psi_n(x)|^2$')
ax.set_title('Open (OBC) — Ground state and first two excited states at N=100, t=1')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('wavefunctions_open.png', dpi=150)
plt.show()