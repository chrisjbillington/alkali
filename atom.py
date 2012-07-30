from __future__ import division
from pylab import *
from wigner import Wigner3j, Wigner6j
import pickle

hbar = 1.054572e-34
c = 2.99792458e8
epsilon_0 = 625000.0/(22468879468420441.0*pi) 
e = 1.6021765e-19
a_0 = 5.29177209e-11
d_B = a_0*e
mu_B = 9.27401e-24


def get_gF(F,I,J,gI,gJ):
    return gJ*(F*(F+1) - I*(I+1) + J*(J+1))/(2*F*(F+1)) + gI*(F*(F+1) + I*(I+1) - J*(J+1))/(2*F*(F+1))

def dipole_moment_zero_field(F, m_F, Fprime, m_Fprime, q, J, Jprime, I, lifetime, omega_0):
    """ Calculates the transition dipole moment in SI units (Coulomb
    metres) for given initial and final F,m_F states of a hydrogenic
    atom. Also required are the initial and final J's, the nuclear spin,
    and the lifetime and angular frequency of the transition."""

    reduced_dipole_moment_J = sqrt(3*pi*epsilon_0 * hbar * c**3 / (lifetime * omega_0**3) * (2*Jprime + 1)/ (2*J + 1))
    reduced_dipole_moment_F = reduced_dipole_moment_J * (-1)**(Fprime + J + 1 + I) * sqrt((2*Fprime+1) * (2*J+1))*Wigner6j(J,Jprime,1,Fprime,F,I)
    
    return reduced_dipole_moment_F * (-1)**(Fprime - 1 + m_F) * sqrt(2*F+1) * Wigner3j(Fprime, 1, F, m_Fprime, q, -m_F)
 
 
def outer(list_a,list_b):
    outer_ab = []
    for a in list_a:
        for b in list_b:
            outer_ab.append((a,b))
    return outer_ab
    
    
def eigensystem(A):
    evals, evecsarray = eigh(A) 
    evecslist = [matrix(evecsarray[:,i]) for i in range(len(evals))]
    return evals, evecslist, matrix(evecsarray)


def find_f(eigenval):
    f1 = (-hbar**2 - sqrt(4*eigenval*hbar**2 + hbar**4))/(2*hbar**2)
    f2 = (-hbar**2 + sqrt(4*eigenval*hbar**2 + hbar**4))/(2*hbar**2)
    return int(round(max([f1,f2])))
    
def find_m(eigenval):
    return int(round(eigenval/hbar))
    
def angular_momentum_operators(J):
    statesJ = linspace(-J,J,2*J+1)
    Jp = diag([hbar*sqrt(J*(J+1) - m_j*(m_j + 1)) for m_j in linspace(-J,J-1,2*J)], -1)
    Jm = diag([hbar*sqrt(J*(J+1) - m_j*(m_j - 1)) for m_j in linspace(-J+1,J,2*J)], 1)
    Jx = matrix((Jp + Jm)/2)
    Jy = matrix((Jp - Jm)/(2j))
    Jz = matrix(diag([hbar*m_j for m_j in statesJ]))
    J2 = Jx**2 + Jy**2 + Jz**2
    nJ = 2*J + 1
    basisvecsJ = [transpose(matrix(vec)) for vec in identity(nJ)]
    return Jx, Jy, Jz, J2, nJ, statesJ, basisvecsJ
    

def angular_momentum_product_space(I,J):
    Ix, Iy, Iz, I2, nI, statesI, basisvecsI = angular_momentum_operators(I)
    Jx, Jy, Jz, J2, nJ,statesJ, basisvecsJ = angular_momentum_operators(J)
    nIJ = (2*I+1)*(2*J+1)
    basisvecsIJ = [transpose(matrix(vec)) for vec in identity(nIJ)]
    statesIJ = outer(statesI, statesJ)
    Fx, Fy, Fz = [kron(Ia,identity(nJ)) + kron(identity(nI), Ja) \
                  for Ia, Ja in zip((Ix,Iy,Iz),(Jx,Jy,Jz))]
    F2 = Fx**2 + Fy**2 + Fz**2
    return Fx, Fy, Fz, F2, nIJ, statesIJ, basisvecsIJ
    
 
class AtomicState(object):
    
    def __init__(self, I, J, gI, gJ, Ahfs, Bhfs=0, Bmax_crossings=500e-4, nB_crossings=5000):
        self.I = I
        self.J = J
        self.Bmax_crossings = Bmax_crossings
        self.nB_crossings = nB_crossings
        Ix, Iy, Iz, I2, nI, statesI, basisvecsI = angular_momentum_operators(I)
        Jx, Jy, Jz, J2, nJ, statesJ, basisvecsJ = angular_momentum_operators(J)  
        self.Fx, self.Fy, self.Fz, F2, nIJ, statesIJ, basisvecsIJ = angular_momentum_product_space(I,J)
        self.H_hfs  = Ahfs * (kron(Ix,Jx) + kron(Iy,Jy) + kron(Iz,Jz))/hbar**2
        if Bhfs != 0:
            # When Bhfs is zero, this term has a division by zero that
            # doesn't get caught. Subsequently multiplying by zero just
            # turns the Infs into Nans. So to avoid getting a H_hfs
            # full of NaNs, we have to skip over this term when it
            # doesn't apply.
            self.H_hfs += Bhfs * (3*((kron(Ix,Jx) + kron(Iy,Jy) + kron(Iz,Jz))/hbar**2)**2 +
                                  3/2 * (kron(Ix,Jx) + kron(Iy,Jy) + kron(Iz,Jz))/hbar**2 - 
                                  I*(I+1)*J*(J+1)*identity(nIJ)) / \
                                 (2*I*(2*I-1)*J*(2*J-1))
        self.mu_x = -(gI*kron(Ix,identity(nJ)) + gJ*kron(identity(nI),Jx)) * mu_B / hbar
        self.mu_y = -(gI*kron(Iy,identity(nJ)) + gJ*kron(identity(nI),Jy)) * mu_B / hbar     
        self.mu_z = -(gI*kron(Iz,identity(nJ)) + gJ*kron(identity(nI),Jz)) * mu_B / hbar
        evalsF, evecsF, S = eigensystem(F2)
        self.flist = sorted(map(find_f, evalsF))
        self.fingerprint = ''.join([str(x) for x in [I,J,gI,gJ,Ahfs,Bhfs,Bmax_crossings,nB_crossings]])
        try:
            self.crossings = pickle.load(open('crossings_'+self.fingerprint+'.pickle'))
            self.crossings_found = True
        except:
            self.crossings_found = False
            self.crossings = []
        
    def Htot(self,B_z):
        return self.H_hfs - self.mu_z*B_z
        
        
    def find_crossings(self):
        B_range = linspace(0,self.Bmax_crossings,self.nB_crossings) 
        evals = [sorted(real(eigensystem(self.Htot(B_range[0]))[0])),
                 sorted(real(eigensystem(self.Htot(B_range[1]))[0]))]
        self.crossings = []
        self.crossingy = []
        self.crossingx = []
        for Bz in B_range[2:]:
            predicted = 2*array(evals[-1]) - array(evals[-2])
            new_evals = array(sorted(eigensystem(self.Htot(Bz))[0]))
            for field, (a,b) in self.crossings:
                new_evals[a], new_evals[b] = new_evals[b], new_evals[a]
            prediction_failures = []
            for i, val in enumerate(new_evals):
                bestmatch = min(abs(val - predicted))
                if not bestmatch == abs(val - predicted[i]):
                    prediction_failures.append(i)
            if prediction_failures:
                self.crossings.append((Bz-0.5*(B_range[1]-B_range[0]), prediction_failures))
                a,b = prediction_failures
                self.crossingx.extend([Bz,Bz])
                self.crossingy.extend([new_evals[a], new_evals[b]])
                new_evals[a], new_evals[b] = new_evals[b], new_evals[a]
            evals.append(new_evals)  
            self.crossings_found = True
            pickle.dump(self.crossings,open('crossings_'+self.fingerprint+'.pickle','wb'))
            
            
    def energy_eigenstates(self, Bz):
        print type(Bz)
        if not self.crossings_found:
            self.find_crossings()
        try:
            results = []
            for B in Bz:
                results.append(self.energy_eigenstates(B))
            vals, alphalist, mlist, evecs = zip(*results)
            vals = tuple(array(vals).transpose())
            evecs = tuple(array(evecs).transpose())
            return vals, alphalist[0], mlist[0], evecs
        except TypeError:
            # Object is not iterable, proceed assuming its a single number:
            pass
        if Bz == 0:
            # Degenerate eigenstates at zero field make it impossible
            # to calculate simultaneous eigenstates of both Htot and
            # Fz. We'll lift that degeneracy just a little bit. This
            # only affects the accuracy of the energy eigenvalues in
            # the 13th decimal place -- far beyond the accuracy of any
            # of these calculations.
            Bz = 1e-15
        evals, evecs, S = eigensystem(self.Htot(Bz))
        mlist=[int(round(real(det(evec.H*self.Fz*evec))/hbar)) for evec in evecs]
        # Sort by energy eigenvalues. m and the eigenvectors are in
        # there so that they get sorted too, though their values aren't
        # being compared (since energies aren't degenerate -- we lifted
        # the degeneracy).
        sortinglist = zip(evals,mlist, evecs)
        sortinglist.sort()
        # Now to apply some swapping to account for crossings at lower
        # fields than we're at. The states will then be sorted by the
        # energy eigenvalues that they converge to at low but nonzero
        # field.
        for field, (a,b) in self.crossings:
            if Bz > field:
                sortinglist[a], sortinglist[b] = sortinglist[b],sortinglist[a]
        # This is Python idiom for unzipping:
        evals, mlist, evecs = zip(*sortinglist)
        # Now here's a list of the F values that these states converge
        # to at low field. Most people call them alpha, or gamma or
        # something like that. They are useful for labeling the states
        # even though they are not eigenvalues of anything.
        alphalist = self.flist
        return evals, alphalist, mlist, evecs

    def rf_transition_matrix_element(self, alpha, m, alphaprime, mprime, direction, Bz):
        evals, alphalist, mlist, evecs = self.energy_eigenstates(Bz)
        for this_alpha, this_m, vec in zip(alphalist,mlist,evecs):
            if this_alpha == alpha and this_m == m:
                initial_state = vec[:]
            if this_alpha == alphaprime and this_m == mprime:
                final_state = vec[:]
        magnetic_moments = {'x': self.mu_x, 'y':self.mu_y, 'z': self.mu_z}
        mu = magnetic_moments[direction]
        return det(final_state.H*mu*initial_state)
        
class AtomicLine(object):
    
    def __init__(self,groundstate,excited_state,omega_0,lifetime):
        self.groundstate = groundstate
        self.excited_state = excited_state
        self.J = groundstate.J
        self.Jprime = excited_state.J
        self.I = groundstate.I
        self.omega_0 = omega_0
        self.lifetime = lifetime
        self.linewidth = 1/lifetime
    
        ground_energies, alphalist, mlist, evecs = self.groundstate.energy_eigenstates(0)
        self.ground_sublevels = zip(['S%d/2'%int(2*self.J)]*len(alphalist), alphalist, mlist)
        excited_energies, alphalist, mlist, evecs = self.excited_state.energy_eigenstates(0)
        self.excited_sublevels = zip(['P%d/2'%int(2*self.Jprime)]*len(alphalist), alphalist, mlist)
        
        self.transitions = {}
        
        self.linewidths = array([0]*len(self.ground_sublevels) + [self.linewidth]*len(self.excited_sublevels))
        
    def get_transitions(self,Bz):
        if Bz in self.transitions:
            return self.transitions[Bz]
        ground_energies, alphalist, mlist, evecs = self.groundstate.energy_eigenstates(Bz)
        excited_energies, alphalist, mlist, evecs = self.excited_state.energy_eigenstates(Bz)
        ground = zip(*zip(*self.ground_sublevels)[1:]) # This just bumps the J's out of the list
        excited = zip(*zip(*self.ground_sublevels)[1:])
        transitions = outer(self.ground_sublevels,self.excited_sublevels)
        energies = outer(ground_energies,excited_energies)
        for transition, energy_pair in zip(transitions[:],energies[:]):
            (J,alpha,m),(Jprime,alphaprime,mprime) = transition
            E,Eprime = energy_pair
            if abs(mprime-m) > 1:
                index = transitions.index(transition)
                del transitions[index]
                del energies[index]
                
        omegas = tuple([(Eprime-E)/hbar for E, Eprime in energies])
        result  = dict([((transition[0],transition[1]),omega) for transition, omega in zip(transitions,omegas)])
        self.transitions[Bz] = result # cache for future speed ups
        return result
        
        
    def transition_dipole_moment(self, J, alpha, m, Jprime, alphaprime, mprime, q, Bz):
        
        evals, alphalist, mlist, evecs = self.groundstate.energy_eigenstates(Bz)
        for this_alpha, this_m, vec in zip(alphalist,mlist,evecs):
            if this_alpha == alpha and this_m == m:
                initial_state = vec[:]
                
        evals, alphalist, mlist, evecs = self.excited_state.energy_eigenstates(Bz)
        for this_alpha, this_m, vec in zip(alphalist,mlist,evecs):
            if this_alpha == alphaprime and this_m == mprime:
                final_state = vec[:]
        
        dipole_moment = 0
        for E, F, mF, ket in zip(*self.groundstate.energy_eigenstates(0)):
            for Eprime, Fprime, mFprime, ketprime in zip(*self.excited_state.energy_eigenstates(0)):
                if (mF - mFprime != q) or abs(Fprime - F) > 1:
                    continue
                initial_projection = det(initial_state.H*ket)
                strength = dipole_moment_zero_field(F, mF, Fprime, mFprime, q, self.J, 
                                                   self.Jprime, self.I, self.lifetime,self.omega_0)
                final_projection = det(ketprime.H*final_state)
                dipole_moment += initial_projection*strength*final_projection
        
        return real(dipole_moment)
            
    def detuning(self, J, alpha, m, Jprime, alphaprime, mprime, intensity, target_scattering_rate, Bz):
        E = sqrt(2*intensity/(c*epsilon_0))
        dipole_moment = self.transition_dipole_moment(J, alpha, m, Jprime, alphaprime, mprime, m-mprime, Bz)
        delta = sqrt(E**2 * dipole_moment**2 / (2*hbar**2) * (self.linewidth/(2*target_scattering_rate) - 1) - self.linewidth**2/4)
        return delta


class FineStructureLine(object):
    def __init__(self, line1, line2):
        self.line1 = line1
        self.line2 = line2
        self.ground_sublevels = line1.ground_sublevels
        self.excited_sublevels = line1.excited_sublevels + line2.excited_sublevels
        self.linewidths = array([0]*len(self.ground_sublevels) + [line1.linewidth]*len(line1.excited_sublevels) + [line2.linewidth]*len(line2.excited_sublevels))
        
    def get_transitions(self,Bz): 
        transitions = {}
        transitions.update(self.line1.get_transitions(Bz))
        transitions.update(self.line2.get_transitions(Bz))
        return transitions
        
    def transition_dipole_moment(self, J, alpha, m, Jprime, alphaprime, mprime, q, Bz):
        if Jprime - J == self.line1.Jprime - self.line1.J:
            return self.line1.transition_dipole_moment(J, alpha, m, Jprime, alphaprime, mprime, q, Bz)
        elif Jprime - J == self.line2.Jprime - self.line2.J:
            return self.line2.transition_dipole_moment(J, alpha, m, Jprime, alphaprime, mprime, q, Bz)
        else:
            raise ValueError('Given J and Jprime do not match any of this line\'s transitions.')


class Laser(object):
    def __init__(self, omega, I, polarisation, line):
        self.omega = omega
        self.polarisation = polarisation
        self.q = {'sigma plus': -1, 'sigma minus': 1, 'pi': 0}[polarisation]
        self.line = line
        self.deltaJ = line.Jprime - line.J
        
        if not callable(I):
            self.I = lambda t: I
        else:
            self.I = I 
                 
                  
class Simulation(object):
    
    def __init__(self,atomic_line,Bz,lasers, dv_dt, delta_v):
        self.atomic_line = atomic_line
        states = atomic_line.ground_sublevels + atomic_line.excited_sublevels
        psi = zeros(len(states),dtype=complex)
        x = 0
        v = 0
        t = 0
        
        self.dipole_moments = zeros((len(states),len(states)))
        field_amplitude_mask = [zeros((len(states),len(states))) for laser in lasers]
        detunings= [zeros((len(states),len(states))) for laser in lasers]
        
        self.stopping = False

        for ((J,alpha,m),(Jprime,alphaprime,mprime)), omega in atomic_line.get_transitions(Bz).items():
            i = states.index((J, alpha, m))
            j = states.index((Jprime, alphaprime, mprime))
            self.dipole_moments[i,j] = self.dipole_moments[j,i] = atomic_line.transition_dipole_moment(int(J[1])/2,alpha, m, int(Jprime[1])/2,alphaprime, mprime, q=m-mprime, Bz=Bz)
            for k, laser in enumerate(lasers):
                if m-mprime == laser.q and int(Jprime[1])/2 - int(J[1])/2 == laser.deltaJ:
                    print k, states[i], states[j]
                    field_amplitude_mask[k][i,j] = field_amplitude_mask[k][j,i] = 1
                    detunings[k][i,j] = laser.omega - omega
                    detunings[k][j,i] = omega - laser.omega # This is actually -1 times the detuning.
             
        def dpsi_dt(x,t,psi):
            coefficients = zeros((len(psi),len(psi)),dtype=complex)
            for i, laser in enumerate(lasers):
                coefficients += sqrt(2*laser.I(x)/(c*epsilon_0))*field_amplitude_mask[i]*exp(1j*detunings[i]*t)
            result = 1j/(2*hbar)*dot(coefficients*self.dipole_moments,psi)
            return result
            
        self.states = states
        self.psi = psi
        self.x = x
        self.v = v
        self.t = t
        self.dpsi_dt = dpsi_dt
        self.dv_dt = dv_dt
        self.delta_v = delta_v
        self.sum_of_decay_probs = 0
        
    def spontaneous_emission(self,i,t):
        individual_decay_probabilities = abs(self.psi)**2 * self.atomic_line.linewidths/(2*pi)*self.dt
        overall_decay_probability = sum(individual_decay_probabilities)
        self.sum_of_decay_probs += overall_decay_probability
        if rand() < overall_decay_probability:
            print 'emission!'
            #Pick an excited state:
            excited_state_index = searchsorted(cumsum(individual_decay_probabilities/overall_decay_probability), rand())
            # Randomly pick a groundstate, weighted by their coupling
            # strengths to the chosen excited state:
            dipole_moments = self.dipole_moments[excited_state_index,:]
            relative_coupling_strengths = dipole_moments**2/dot(dipole_moments,dipole_moments)
            groundstate_index = searchsorted(cumsum(relative_coupling_strengths),random())
            # Set everything to zero:
            self.psi[:] = 0
            # Put all the population in the chosen groundstate:
            self.psi[groundstate_index] = 1   
            # Give the atom a photon recoil in a random direction:
            self.v += (random_integers(0,1)*2 - 1)*self.delta_v
            return excited_state_index, groundstate_index
    
    def stop(self):
        self.stopping = True
                            
    def run(self, dt, triggers={}, repeated_triggers={},emission_trigger=None):
        self.dt = dt
        t = self.t
        i = 0
        while True:
            if self.stopping:
                self.stopping = False
                break
            if isnan(self.psi).any() or isinf(self.psi).any() or \
               isnan(self.x) or isinf(self.x) or \
               isnan(self.v) or isinf(self.v) :
                raise OverflowError('It exploded :(')
                
            k1_psi = self.dpsi_dt(self.x, t, self.psi)
            k1_v = self.dv_dt(self.x, self.psi)
            k1_x = self.v
            
            k2_psi = self.dpsi_dt(self.x + 0.5*dt*k1_x, t + 0.5*dt, self.psi + 0.5*dt*k1_psi)
            k2_v = self.dv_dt(self.x + 0.5*dt*k1_x, self.psi + 0.5*dt*k1_psi)
            k2_x = self.v + 0.5*dt*k1_v
            
            k3_psi = self.dpsi_dt(self.x + 0.5*dt*k2_x, t + 0.5*dt, self.psi + 0.5*dt*k2_psi)
            k3_v = self.dv_dt(self.x + 0.5*dt*k2_x, self.psi + 0.5*dt*k2_psi)
            k3_x = self.v + 0.5*dt*k2_v
            
            k4_psi = self.dpsi_dt(self.x + dt*k3_x, t + dt, self.psi + dt*k3_psi)
            k4_v = self.dv_dt(self.x + dt*k3_x, self.psi + dt*k3_psi)
            k4_x = self.v + dt*k3_v
            
            self.psi += dt/6*(k1_psi + 2*k2_psi + 2*k3_psi + k4_psi) 
            self.v += dt/6*(k1_v + 2*k2_v + 2*k3_v + k4_v) 
            self.x += dt/6*(k1_x + 2*k2_x + 2*k3_x + k4_x) 
            
            emission = self.spontaneous_emission(i,t)
            if emission and emission_trigger:
                emission_trigger(i,t,*emission)
            
            for trigger in triggers:
                if i == triggers:
                    triggers[i](i,t,self.psi,x,v)  
                             
            for trigger in repeated_triggers:
                if i%trigger == 0:
                    repeated_triggers[trigger](i/trigger,t,self.psi,self.x,self.v) 
                    
            t += dt
            i += 1

 


