from __future__ import division
from numpy import *
from alkali.rubidium87 import *
from uncertainties import ufloat

# Zeeman splitting of states specified by a list e.g. ['a', 'b', ..]
def splitting(states, B):
    # average splitting between states in Hz for B in Gauss
    state_ints = [ord(x)-ord('a') for x in sort(states)]
    energies = array(rubidium_87_S12_state.energy_eigenstates(B*1e-4)[0])
    Ediff = mean(energies[state_ints][1:]-energies[state_ints][:-1])/(2*pi*hbar)
    return Ediff

# Solve for field from a given frequency
from scipy.optimize import fsolve
from scipy.misc import derivative

def solve_field(states, frequency, u_frequency=0, dB=1e-3):
    zero_field_splitting = splitting(states, 0)
    B0 = (frequency-zero_field_splitting)/derivative(lambda B: splitting(states, B), dB, dx=dB)
    B = fsolve(lambda B: splitting(states, B) - frequency, B0)[0]
    u_B = abs(u_frequency/derivative(lambda B: splitting(states, B), B, dx=dB))
    return B, u_B

def format_unc(x, u_x, m=1):
    format_string = '{:.%iuS}' % m
    return format_string.format(ufloat(x, abs(u_x)))

if __name__ == '__main__':
    # Parameters
    state_list = ['b', 'c']          # |F=1,m=0> and |F=1,m=-1>
    # state_list = ['a', 'b', 'c']   # use this to compute average Zeeman splitting of F = 1 ground states
    # state_list = ['c', 'f']        # |F=1,m=-1> and |F=2,m=+1>

    # Compute the transition energy at 100 G and 165 G
    print splitting(state_list, 100)
    print splitting(state_list, 165)
    # 70910180.9881
    # 117639118.382

    # Find the magnetic field for a transition frequency of 700kHz (with an uncertainty of 1 Hz)
    f0, u_f0 = 700e3, 1
    Bsol, u_Bsol = solve_field(state_list, 700e3, 1)
    print format_unc(Bsol, u_Bsol)
    # 0.996526(1)

    # Find the magnetic field for transition frequencies listed by ANU
    f_list = array([117.649, 118.222, 120.343, 117.209])*1e6
    B_list = [solve_field(state_list, f)[0] for f in f_list]
    print B_list
    # [165.01368203711371, 165.807012590174, 168.74286121992532, 164.40443627607195]

    # ... ditto but now compute uncertainty in inferred field
    u_f = 500
    B_list_u = [solve_field(state_list, f, u_f) for f in f_list]
    for f, B in zip(f_list, B_list_u):
        print 'f0 = %s MHz, B = %s G' % (format_unc(f/1e6, u_f/1e6), format_unc(*B))
    # f0 = 117.6490(5) MHz, B = 165.0137(7) G
    # f0 = 118.2220(5) MHz, B = 165.8071(7) G
    # f0 = 120.3430(5) MHz, B = 168.7429(7) G
    # f0 = 117.2090(5) MHz, B = 164.4045(7) G      