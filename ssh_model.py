import numpy as np
import matplotlib.pyplot as plt


def ssh_hamiltonian(N, v, w=1.0, bc="open"):
    if bc not in ("open", "periodic"):
        raise ValueError("bc must be 'open' or 'periodic'")
    dim = 2 * N
    H = np.zeros((dim, dim), dtype=float)
    for n in range(N):
        a_n, b_n = 2 * n, 2 * n + 1
        H[b_n, a_n] = H[a_n, b_n] = -v
        if n < N - 1:
            a_np1 = 2 * (n + 1)
            H[a_np1, b_n] = H[b_n, a_np1] = -w
        elif bc == "periodic":
            H[0, b_n] = H[b_n, 0] = -w
    return H


def add_disorder(H, N, W, rng):
    H_dis = H.copy()
    for n in range(N):
        H_dis[2 * n,     2 * n    ] += rng.uniform(-W, W)
        H_dis[2 * n + 1, 2 * n + 1] += rng.uniform(-W, W)
    return H_dis


def localize_pair(psi0, psi1):
    """Rotate a pair of edge states into the maximally left/right localized basis.

    Diagonalizes the left-minus-right weight operator within the two-state
    subspace, which gives the optimal rotation for both degenerate and
    slightly-split pairs.
    """
    mid = len(psi0) // 2
    sigma = np.ones(len(psi0))
    sigma[mid:] = -1.0
    M = np.array([
        [np.dot(psi0, sigma * psi0), np.dot(psi0, sigma * psi1)],
        [np.dot(psi1, sigma * psi0), np.dot(psi1, sigma * psi1)],
    ], dtype=float)
    _, vecs = np.linalg.eigh(M)
    left  = vecs[0, 1] * psi0 + vecs[1, 1] * psi1
    right = vecs[0, 0] * psi0 + vecs[1, 0] * psi1
    if np.sum(np.abs(left[:mid])**2) < np.sum(np.abs(left[mid:])**2):
        left, right = right, left
    return left, right


w = 1.0
ratios = [0.5, 1.0, 2.0]
N_bulk = 100
N_edge = 50

phase_labels = {
    0.5: 'Topological insulator  ($v < w$)',
    1.0: 'Critical point  ($v = w$)',
    2.0: 'Trivial insulator  ($v > w$)',
}

# ── Task 2: Bulk band structure — one file per v/w ───────────────────────────
k_vals = np.linspace(-np.pi, np.pi, N_bulk, endpoint=False)

for r in ratios:
    v = r * w
    lower, upper = np.empty(N_bulk), np.empty(N_bulk)
    for i, k in enumerate(k_vals):
        h_k = np.array([[0, -(v + w * np.exp(1j * k))],
                        [-(v + w * np.exp(-1j * k)), 0]])
        e = np.linalg.eigvalsh(h_k)
        lower[i], upper[i] = e[0], e[1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_vals / np.pi, lower, 'C0', lw=2, alpha=0.9, label='Lower band')
    ax.plot(k_vals / np.pi, upper, 'C1', lw=2, alpha=0.9, ls='--', label='Upper band')
    ax.fill_between(k_vals / np.pi, lower, upper, alpha=0.12, color='gray', label='Gap')
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_xlabel(r'$k/\pi$')
    ax.set_ylabel('Energy $E$')
    ax.set_title(f'SSH band structure  ($v/w={r}$,  $w=1$,  $N={N_bulk}$)')
    ax.text(0.03, 0.96, phase_labels[r], transform=ax.transAxes,
            fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'band_structure_v{str(r).replace(".", "p")}.png', dpi=150)
    plt.show()

# ── Task 3a: OBC energy spectrum — one file per v/w ──────────────────────────
for r in ratios:
    v = r * w
    eigs = np.linalg.eigvalsh(ssh_hamiltonian(N_edge, v, w, bc="open"))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(range(len(eigs)), eigs, s=14, alpha=0.7, zorder=3)
    ax.axhline(0, color='r', lw=0.8, ls='--', label='$E=0$')
    ax.set_xlabel('State index')
    ax.set_ylabel('Energy $E$')
    ax.set_title(f'SSH OBC spectrum  ($v/w={r}$,  $N={N_edge}$,  $w=1$)')
    ax.legend()
    ax.grid(alpha=0.3)
    if r == 1.0:
        ax.text(0.5, 0.5, 'Gapless\ncritical point', transform=ax.transAxes,
                fontsize=13, va='center', ha='center', color='darkred', alpha=0.85,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))
    plt.tight_layout()
    plt.savefig(f'obc_spectrum_v{str(r).replace(".", "p")}.png', dpi=150)
    plt.show()

# ── Task 3c: Zero-energy edge-state wavefunctions (v/w=0.5) ──────────────────
v = 0.5 * w
eigs, evecs = np.linalg.eigh(ssh_hamiltonian(N_edge, v, w, bc="open"))
zero_idx = np.argsort(np.abs(eigs))[:2]
x = np.arange(2 * N_edge)

psi_left, psi_right = localize_pair(evecs[:, zero_idx[0]], evecs[:, zero_idx[1]])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, np.abs(psi_left)**2,  'C0', ls='-',  marker='o', ms=3, lw=1.8, alpha=0.90, label='Left edge state')
ax.plot(x, np.abs(psi_right)**2, 'C1', ls='--', marker='s', ms=3, lw=1.8, alpha=0.90, label='Right edge state')
ax.set_xlabel('Site index')
ax.set_ylabel(r'$|\psi(x)|^2$')
ax.set_title('Zero-energy edge states (OBC, $v/w=0.5$, $N=50$)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('edge_wavefunctions.png', dpi=150)
plt.show()

# ── Task 4: Disorder robustness — one spectrum + one wavefunction file per W ──
rng = np.random.default_rng(42)
v = 0.5 * w
W_vals = [0.2, 0.4]
H_base = ssh_hamiltonian(N_edge, v, w, bc="open")

for W in W_vals:
    H_dis = add_disorder(H_base, N_edge, W, rng)
    eigs_d, evecs_d = np.linalg.eigh(H_dis)
    zero_idx_d = np.argsort(np.abs(eigs_d))[:2]
    psi_l, psi_r = localize_pair(evecs_d[:, zero_idx_d[0]], evecs_d[:, zero_idx_d[1]])
    tag = str(W).replace('.', 'p')

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(range(len(eigs_d)), eigs_d, s=14, alpha=0.7, zorder=3)
    ax.axhline(0, color='r', lw=0.8, ls='--', label='$E=0$')
    ax.set_xlabel('State index')
    ax.set_ylabel('Energy $E$')
    ax.set_title(f'Disordered OBC spectrum ($W={W}$, $v/w=0.5$, $N={N_edge}$)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'disorder_spectrum_W{tag}.png', dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, np.abs(psi_l)**2, 'C0', ls='-',  marker='o', ms=3, lw=1.8, alpha=0.90, label='Left edge state')
    ax.plot(x, np.abs(psi_r)**2, 'C1', ls='--', marker='s', ms=3, lw=1.8, alpha=0.90, label='Right edge state')
    ax.set_xlabel('Site index')
    ax.set_ylabel(r'$|\psi(x)|^2$')
    ax.set_title(f'Disordered edge wavefunctions ($W={W}$, $v/w=0.5$, $N={N_edge}$, $w=1$)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'disorder_wavefunctions_W{tag}.png', dpi=150)
    plt.show()
