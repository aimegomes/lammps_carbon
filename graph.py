import numpy as np
import matplotlib.pyplot as plt

# Simple plot of just the real part
data = np.loadtxt('data/S_q_omega_clean_q100.txt')
time = data[:, 0]
F_qt_real = data[:, 1]
F_qt_im = data[:, 2]

plt.figure(figsize=(10, 4))
plt.plot(time, F_qt_real, 'b-', linewidth=1)
plt.xlabel('omega')
plt.ylabel('Real(S(omega,q))')
plt.title('Scattering Function - Real Part for q=100')
#plt.grid(True, alpha=0.3)
plt.savefig('F_real_q100.png')
plt.clf()
plt.figure(figsize=(10, 4))
plt.plot(time, F_qt_im, 'b-', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Im(F(q,t))')
plt.title('Intermediate Scattering Function - Imaginary Part for q=100')
#plt.grid(True, alpha=0.3)
plt.savefig('F_im_q100.png')
