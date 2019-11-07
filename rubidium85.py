from __future__ import division
from pylab import *

from .atom import AtomicState, AtomicLine, FineStructureLine, hbar, c, epsilon_0, e, d_B
       
# 85Rb D line properties
# data from Steck, "Rubidium 85 D Line Data," revision 2.1.6, 20 September 2013
# http://steck.us/alkalidata

m_Rb85 = 1.409993199e-25
m = m_Rb85

# Frequencies and linewidths: 
f0_D2 = 384.230406373e12
omega0_D2 = 2*pi*f0_D2
k0_D2 = 2*pi*f0_D2/c
lifetime_D2 = 26.2348e-9
gamma_D2 = 1/lifetime_D2
f0_D1 = 377.107463380e12
omega0_D1 = 2*pi*f0_D1
k0_D1 = 2*pi*f0_D1/c
lifetime_D1 = 27.679e-9
gamma_D1 = 1/lifetime_D1
wavelength = 780.241368271e-9
sigma0 = 2.90669411930e-13 # m^2

# Hyperfine structure constants:
A_S12 = 2*pi*hbar*1.0119108130e9
A_P12=2*pi*hbar*120.527e6
A_P32 = 2*pi*hbar*25.0020e6
B_P32 = 2*pi*hbar*25.790e6

# Lande g-factors:
gI = -0.00029364000
gJ_S12 = 2.00233113
gJ_P12 = 0.666
gJ_P32 = 1.3362
 
rubidium_85_S12_state = AtomicState(I=5/2, J=1/2, gI=gI, gJ=gJ_S12, Ahfs=A_S12)
rubidium_85_P12_state = AtomicState(I=5/2, J=1/2, gI=gI, gJ=gJ_P12, Ahfs=A_P12)
rubidium_85_P32_state = AtomicState(I=5/2, J=3/2, gI=gI, gJ=gJ_P32, Ahfs=A_P32, Bhfs=B_P32)

rubidium_85_D1_line = AtomicLine(
    rubidium_85_S12_state, rubidium_85_P12_state, omega0_D1, lifetime_D1
)
rubidium_85_D2_line = AtomicLine(
    rubidium_85_S12_state, rubidium_85_P32_state, omega0_D2, lifetime_D2
)

rubidium_85_D_line = FineStructureLine(
    rubidium_85_D1_line, rubidium_85_D2_line, N=5, L=0, Nprime=5, Lprime=1
)

if __name__ == '__main__':
    import time

    # Example:
    Bz = 34e-4
    transition_frequencies = rubidium_85_D_line.transitions(Bz)
    dipole_moment_1 = rubidium_85_D_line.transition_dipole_moment(
        1 / 2, 2, -2, 1 / 2, 2, -2, Bz
    )
    dipole_moment_2 = rubidium_85_D_line.transition_dipole_moment(
        1 / 2, 3, -1, 1 / 2, 3, -1, Bz
    )


    # Another example:
    Bz = linspace(0, 200e-4, 1000)
    eigenstates = rubidium_85_P32_state.energy_eigenstates(Bz)

    for (alpha, mF), (E, psi) in eigenstates.items():
        plot(
            Bz * 1e4, E / (2 * pi * hbar * 1e6), label=r'$|%d, %d\rangle$' % (alpha, mF)
        )
    grid(True)
    xlabel('B (Gauss)')
    ylabel('E (MHz)')
    legend()
    show()










    
