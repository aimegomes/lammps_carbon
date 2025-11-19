import numpy as np
import MDAnalysis as mda
import os
import matplotlib.pyplot as plt
from scipy.signal import correlate

def to_array(u, group):
    """Extract position and velocity"""
    x_matrix = np.zeros((u.trajectory.n_frames, group.n_atoms, 3))
    for ts in u.trajectory:
        x_matrix[ts.frame] = group.positions
    return x_matrix

# MAIN CODE
q_value = 3
dt = 0.001

print(f"Starting calculation for q = {q_value}")
os.makedirs("data", exist_ok=True)

v = mda.Universe("mdaconfig.data", dt=dt)
v.load_new("dump.lammps", format="LAMMPSDUMP")

Lx = v.dimensions[0]
print(f"Box dimension Lx: {Lx}")

qx = 2 * np.pi * q_value / Lx

carbon_atoms = v.select_atoms("type 1")
N = carbon_atoms.n_atoms
print(f"Selected {N} carbon atoms")

positions = to_array(v, carbon_atoms)
n_frames = 2000
print(f"Loaded {n_frames} frames")

print("Computing F(q,t) using density fluctuation method...")

#Density fluctuations \rho(q,t) = \sum exp(iq·r_{j}(t))
rho_qt = np.zeros(n_frames, dtype=complex)

for t in range(n_frames):
    # Sum over single index only - O(N) instead of O(N²)
    for j in range(N):
        rho_qt[t] += np.exp(1j * qx * positions[t, j, 0])
    
    if t % 1000 == 0:
        print(f"Processed frame {t}/{n_frames}")

# F(q,t) is the auto-correlation of density fluctuations
# F(q,t) = (1/N) ⟨\rho(q,t) \rho(-q,0)⟩
correlation = correlate(rho_qt, np.conj(rho_qt), mode='full')
correlation = correlation[len(correlation)//2:]  # Take second half
normalization = np.arange(len(correlation), 0, -1)
F_qt = correlation / (N * normalization)

# Save only first half to avoid noise
n_save = min(2000, len(F_qt))  # Save only first 2000 points or less
output_file_F = f"data/F_qt_clean_q{q_value}.txt"
np.savetxt(output_file_F,
           np.column_stack([
               np.arange(n_save) * dt,
               np.real(F_qt[:n_save]),
               np.imag(F_qt[:n_save])
           ]),
           header="Time Real(F(q,t)) Imag(F(q,t))")

print(f"Clean F(q,t) saved to: {output_file_F}")

print("Computing S(q,ω) via Fourier transform...")

# Compute S(q,ω) = (1/2π) ∫ F(q,t) exp(iωt) dt
# Use the clean F(q,t) we just computed
F_qt_clean = F_qt[:n_save]

# Zero-padding for better frequency resolution
F_qt_padded = np.pad(F_qt_clean, (0, n_save), mode='constant')

# Fourier transform to get S(q,ω)
S_qw = np.fft.fft(F_qt_padded) * dt / (2 * np.pi)

# Frequencies (ω)
frequencies = np.fft.fftfreq(2 * n_save, d=dt)

# Take only positive frequencies
positive_idx = frequencies >= 0
frequencies_positive = frequencies[positive_idx]
S_qw_positive = S_qw[positive_idx]

# Save S(q,ω)
output_file_S = f"data/S_q_omega_clean_q{q_value}.txt"
np.savetxt(output_file_S,
           np.column_stack([
               frequencies_positive,
               np.real(S_qw_positive),
               np.imag(S_qw_positive),
               np.abs(S_qw_positive)  # Also save magnitude
           ]),
           header="Frequency Real(S(q,ω)) Imag(S(q,ω)) Magnitude(S(q,ω))")

print(f"S(q,ω) saved to: {output_file_S}")

# Plot both F(q,t) and S(q,ω)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot F(q,t)
ax1.scatter(np.arange(n_save) * dt, np.real(F_qt_clean), alpha=0.6, s=10)
ax1.set_xlabel('Time')
ax1.set_ylabel('Real(F(q,t))')
ax1.set_title(f'Intermediate Scattering Function F(q,t), q={q_value}')
ax1.grid(True, alpha=0.3)

# Plot S(q,ω) - show magnitude
ax2.plot(frequencies_positive, np.abs(S_qw_positive), 'r-', linewidth=1.5)
ax2.set_xlabel('Frequency ω')
ax2.set_ylabel('|S(q,ω)|')
ax2.set_title(f'Dynamic Structure Factor S(q,ω), q={q_value}')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, min(50, frequencies_positive.max())])  # Limit x-axis for better view

plt.tight_layout()
plt.show()

print("Calculation complete!")
print(f"F(q,t) range: {n_save} time points")
print(f"S(q,ω) range: {len(frequencies_positive)} frequency points")
print(f"Frequency range: 0 to {frequencies_positive.max():.2f}")
