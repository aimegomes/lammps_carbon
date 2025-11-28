#!/bin/bash
/bin/hostname

echo " "
echo "-------------------------------"
echo "Repetitions for different q's"
echo " "

ts -S 7

for Q in `seq 1 100`
do 
    echo "Rodando analises para q = $Q"
    
    # Create unique temporary directory for each Q
    TEMP_DIR="temp_q${Q}"
    mkdir -p "$TEMP_DIR"
    
    cat > "$TEMP_DIR/struc_fact.py" << Fim_entrada
import numpy as np
import MDAnalysis as mda
import os
import plotext as tpl
from scipy.signal import correlate
import sys
import os

def to_array(u, group):
    """Extract position and velocity"""
    x_matrix = np.zeros((u.trajectory.n_frames, group.n_atoms, 3))
    for ts in u.trajectory:
        x_matrix[ts.frame] = group.positions
    return x_matrix

# MAIN CODE
q_value = $Q
dt = 0.001

print(f"Starting calculation for q = {q_value}")
os.makedirs("data", exist_ok=True)

# Use absolute paths to avoid any path issues
config_file = "../mdaconfig.data"
dump_file = "../dump.lammps"

if not os.path.exists(config_file) or not os.path.exists(dump_file):
    print(f"Error: Required files not found!")
    print(f"Looking for: {config_file}, {dump_file}")
    sys.exit(1)

v = mda.Universe(config_file, dt=dt)
v.load_new(dump_file, format="LAMMPSDUMP")

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

# Density fluctuations \\rho(q,t) = \\sum exp(iqrj}(t))
rho_qt = np.zeros(n_frames, dtype=complex)

for t in range(n_frames):
    # Sum over single index only - O(N) instead of O(N²)
    for j in range(N):
        rho_qt[t] += np.exp(1j * qx * positions[t, j, 0])
    
    if t % 1000 == 0:
        print(f"Processed frame {t}/{n_frames}")

# F(q,t) is the auto-correlation of density fluctuations
correlation = correlate(rho_qt, np.conj(rho_qt), mode='full')
correlation = correlation[len(correlation)//2:]  # Take second half
normalization = np.arange(len(correlation), 0, -1)
F_qt = correlation / (N * normalization)

# Save only first half to avoid noise
n_save = min(2000, len(F_qt))  # Save only first 2000 points or less
output_file_F = f"../data/F_qt_clean_q{q_value}.txt"
np.savetxt(output_file_F,
           np.column_stack([
               np.arange(n_save) * dt,
               np.real(F_qt[:n_save]),
               np.imag(F_qt[:n_save])
           ]),
           header="Time Real(F(q,t)) Imag(F(q,t))")

print(f"Clean F(q,t) saved to: {output_file_F}")

print("Computing S(q,ω) via Fourier transform...")

# Compute
# Use the clean F(q,t) we just computed
F_qt_clean = F_qt[:n_save]

# Zero-padding for better frequency resolution
F_qt_padded = np.pad(F_qt_clean, (0, n_save), mode='constant')

# Fourier transform
S_qw = np.fft.fft(F_qt_padded) * dt

# Frequencies
frequencies = np.fft.fftfreq(2 * n_save, d=dt)

# Take only positive frequencies
positive_idx = frequencies >= 0
frequencies_positive = frequencies[positive_idx]
S_qw_positive = S_qw[positive_idx]

# Save S
output_file_S = f"../data/S_q_omega_clean_q{q_value}.txt"
np.savetxt(output_file_S,
           np.column_stack([
               frequencies_positive,
               np.real(S_qw_positive),
               np.imag(S_qw_positive),
               np.abs(S_qw_positive)  # Also save magnitude
           ]),
           header="Frequency Real(S(q,ω)) Imag(S(q,ω)) Magnitude(S(q,ω))")

print(f"S(q,ω) saved to: {output_file_S}")

print("Calculation complete for q = {q_value}!")
print(f"F(q,t) range: {n_save} time points")
print(f"S(q,ω) range: {len(frequencies_positive)} frequency points")
print(f"Frequency range: 0 to {frequencies_positive.max():.2f}")
Fim_entrada

    echo "Submitting job for q = $Q"
    ts bash -c "cd '$TEMP_DIR' && python3 struc_fact.py && cd .. && rm -rf '$TEMP_DIR'"

done

echo "All jobs submitted! Monitor with: tsp -l"
