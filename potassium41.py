from __future__ import division
from pylab import *

from atom import AtomicState, AtomicLine, FineStructureLine, Laser, Simulation, hbar, c, epsilon_0, e, d_B
       
# 41K D line properties:

m_K = 40.96182576*1.660539e-27
m = m_K

# Frequencies and linewidths: 
f0_D2 = 391.01640621e12
omega0_D2 = 2*pi*f0_D2
k0_D2 = 2*pi*f0_D2/c
lifetime_D2 = 26.37e-9
gamma_D2 = 1/lifetime_D2
f0_D1 = 389.286294205e12
omega0_D1 = 2*pi*f0_D1
k0_D1 = 2*pi*f0_D1/c
lifetime_D1 = 26.37e-9
gamma_D1 = 1/lifetime_D1
wavelength = 766.70045870e-9
sigma0 = 3*wavelength**2 / (2*pi)

# Hyperfine structure constants:
A_S12 = 2*pi*hbar*127.0069352e6
A_P12=2*pi*hbar*15.245e6
A_P32 = 2*pi*hbar*3.363e6
B_P32 = 2*pi*hbar*3.351e6

# Lande g-factors:
gI = -0.00007790600
gJ_S12 = 2.00229421
gJ_P12 = 2./3
gJ_P32 = 4./3
 
potassium_41_S12_state = AtomicState(I=3/2, J=1/2, gI=gI, gJ=gJ_S12, Ahfs=A_S12)
potassium_41_P12_state = AtomicState(I=3/2, J=1/2, gI=gI, gJ=gJ_P12, Ahfs=A_P12)
potassium_41_P32_state = AtomicState(I=3/2, J=3/2, gI=gI, gJ=gJ_P32, Ahfs=A_P32, Bhfs=B_P32)

potassium_41_D1_line = AtomicLine(potassium_41_S12_state, potassium_41_P12_state, omega0_D1, lifetime_D1)
potassium_41_D2_line = AtomicLine(potassium_41_S12_state, potassium_41_P32_state, omega0_D2, lifetime_D2)

potassium_41_D_line = FineStructureLine(potassium_41_D1_line,potassium_41_D2_line)

if __name__ == '__main__':
    import time
    # Example:
    Bz = 34e-4
    transition_frequencies =  potassium_41_D_line.get_transitions(Bz)
    dipole_moment_1 = potassium_41_D_line.transition_dipole_moment(1/2, 2, -2, 1/2, 2, -2, 0, Bz)
    dipole_moment_2 = potassium_41_D_line.transition_dipole_moment(1/2, 1, -1, 1/2, 1, -1, 0, Bz)

    # Another example:
    Bz = linspace(0,200e-4,1000)
    evals, alphalist, mlist, evecs = potassium_41_S12_state.energy_eigenstates(Bz)
    for eval, alpha, m in zip(evals, alphalist, mlist):
        plot(Bz*1e4, eval/(2*pi*hbar*1e6), label=r'$|%d, %d\rangle$'%(alpha, m))
    grid(True)
    xlabel('B (Gauss)')
    ylabel('E (MHz)')
    legend()
    show()





    
