from __future__ import division
from pylab import *

from .atom import AtomicState, AtomicLine, FineStructureLine, hbar, c, epsilon_0, e, d_B
       
# 87Rb D line properties
# data from Steck, "Rubidium 87 D Line Data," revision 2.1.4, 23 December 2010
# http://steck.us/alkalidata

m_Rb87 = 1.443160648e-25
m = m_Rb87

# Frequencies and linewidths: 
f0_D2 = 384.2304844685e12
omega0_D2 = 2*pi*f0_D2
k0_D2 = 2*pi*f0_D2/c
lifetime_D2 = 26.2348e-9
gamma_D2 = 1/lifetime_D2
f0_D1 = 377.107463380e12
omega0_D1 = 2*pi*f0_D1
k0_D1 = 2*pi*f0_D1/c
lifetime_D1 = 27.679e-9
gamma_D1 = 1/lifetime_D1
wavelength = 780.241209686e-9
sigma0 = 2.906692937721e-13 # m^2

# Hyperfine structure constants:
A_S12 = 2*pi*hbar*3.417341305452145e9
A_P12=2*pi*hbar*407.24e6
A_P32 = 2*pi*hbar*84.7185e6
B_P32 = 2*pi*hbar*12.4965e6

# Lande g-factors:
gI = -0.0009951414
gJ_S12 = 2.00233113
gJ_P12 = 0.666
gJ_P32 = 1.3362
 
rubidium_87_S12_state = AtomicState(I=3/2, J=1/2, gI=gI, gJ=gJ_S12, Ahfs=A_S12)
rubidium_87_P12_state = AtomicState(I=3/2, J=1/2, gI=gI, gJ=gJ_P12, Ahfs=A_P12)
rubidium_87_P32_state = AtomicState(I=3/2, J=3/2, gI=gI, gJ=gJ_P32, Ahfs=A_P32, Bhfs=B_P32)

rubidium_87_D1_line = AtomicLine(
    rubidium_87_S12_state, rubidium_87_P12_state, omega0_D1, lifetime_D1
)
rubidium_87_D2_line = AtomicLine(
    rubidium_87_S12_state, rubidium_87_P32_state, omega0_D2, lifetime_D2
)

rubidium_87_D_line = FineStructureLine(
    rubidium_87_D1_line, rubidium_87_D2_line, N=5, L=0, Nprime=5, Lprime=1
)

if __name__ == '__main__':
    import time

    # Example:
    Bz = 34e-4
    transition_frequencies = rubidium_87_D_line.transitions(Bz)
    dipole_moment_1 = rubidium_87_D_line.transition_dipole_moment(
        1 / 2, 2, -2, 1 / 2, 2, -2, Bz
    )
    dipole_moment_2 = rubidium_87_D_line.transition_dipole_moment(
        1 / 2, 1, -1, 1 / 2, 1, -1, Bz
    )


    # Another example:
    Bz = linspace(0, 200e-4, 1000)
    eigenstates = rubidium_87_P32_state.energy_eigenstates(Bz)

    for (alpha, mF), (E, psi) in eigenstates.items():
        plot(
            Bz * 1e4, E / (2 * pi * hbar * 1e6), label=r'$|%d, %d\rangle$' % (alpha, mF)
        )
    grid(True)
    xlabel('B (Gauss)')
    ylabel('E (MHz)')
    legend()
    show()








    
