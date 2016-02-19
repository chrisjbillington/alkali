from __future__ import division
from numpy import *
import alkali.rubidium87 as rb87
import alkali.rubidium85 as rb85
from uncertainties import ufloat

ground_level = {'rb85': rb85.rubidium_85_S12_state, 'rb87': rb87.rubidium_87_S12_state}

# Zeeman splitting of states specified by a list e.g. ['a', 'b', ..]
def splitting(states, B, isotope='rb87'):
    # average splitting between states in Hz for B in Gauss
    state_ints = [ord(x)-ord('a') for x in sort(states)]
    energies = array(ground_level[isotope].energy_eigenstates(B*1e-4)[0])
    Ediff = mean(energies[state_ints][1:]-energies[state_ints][:-1])/(2*pi*rb87.hbar)
    return Ediff

# Solve for field from a given frequency
from scipy.optimize import fsolve
from scipy.misc import derivative

def solve_field(states, frequency, u_frequency=0, dB=1e-3, isotope='rb87'):
    zero_field_splitting = splitting(states, 0, isotope)
    B0 = (frequency-zero_field_splitting)/derivative(lambda B: splitting(states, B, isotope), dB, dx=dB)
    B = fsolve(lambda B: splitting(states, B, isotope) - frequency, B0)[0]
    u_B = abs(u_frequency/derivative(lambda B: splitting(states, B, isotope), B, dx=dB))
    return B, u_B

def format_unc(x, u_x, m=1):
    format_string = '{:.%iuS}' % m
    return format_string.format(ufloat(x, abs(u_x)))

if __name__ == '__main__':
    # Parameters
    rb87_states = ['b', 'c']          #  |F=1, m=0> and |F=1,m=-1>
    rb85_states = ['d', 'e']          #  |F=2,m=-1> and |F=2,m=-2>

    # Find the magnetic field for particular transition frequencies
    f_list = array([82.63])*1e6
    B_list = [solve_field(rb85_states, f, isotope='rb85')[0] for f in f_list]
    print B_list
    # [164.86685751724897]

    # ... ditto but now compute uncertainty in inferred field
    u_f = 5e3
    B_list_u = [solve_field(rb85_states, f, u_f, isotope='rb85') for f in f_list]
    for f, B in zip(f_list, B_list_u):
        print 'f0 = %s MHz, B = %s G' % (format_unc(f/1e6, u_f/1e6), format_unc(*B))
    # f0 = 82.630(5) MHz, B = 164.867(9) G