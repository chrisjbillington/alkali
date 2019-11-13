import os
from tempfile import gettempdir
import numpy as np
from .wigner import Wigner3j
import shelve
from tqdm import tqdm
import itertools

pi = np.pi
hbar = 1.054571628e-34
c = 2.99792458e8
mu_0 = 4 * pi * 1e-7
epsilon_0 = 1 / (mu_0 * c ** 2)
e = 1.602176487e-19
a_0 = 5.2917720859e-11
d_B = a_0 * e
mu_B = 9.27400915e-24
gs = 2.0023193043622


def get_gF(F, I, J, gJ, gI):
    """Compute the effective low-field lande-g factor for a state of total angular
    momentum quantum number F, electronic and nuclear angular momentum quantum numbers J
    and I, and electronic and nuclear Lande g factors gJ and gI"""
    term1 = gJ * (F * (F + 1) - I * (I + 1) + J * (J + 1)) / (2 * F * (F + 1))
    term2 = gI * (F * (F + 1) + I * (I + 1) - J * (J + 1)) / (2 * F * (F + 1))
    return term1 + term2


def outer(a, b):
    return list(itertools.product(a, b))


def _int(x):
    """Round x to an integer, returning an int dtype"""
    x = np.round(np.array(x))
    result = np.array(x, dtype=int)
    assert np.all(abs(x - result) < 0.05)
    return result


def _halfint(x):
    """Round x to a half-integer, returning an int if the result is a whole number and a
    float otherwise"""
    x = np.array(x)
    twice_x = np.round(2 * x)
    # Halve and decide whether to return as float or int (array)
    if np.any(twice_x % 2):
        result = twice_x / 2
    else:
        result = np.array(twice_x // 2, dtype=int)
    assert np.all(abs(x - result) < 0.05)
    return result


def _cleanarrays(scale, *arrs):
    """For arrays that have been computed numerically, set elements less than 10 ×
    machine epsilon times `scale` to zero. This essentially rounds to exactly zero
    elements that would be zero if computed exactly, but are not due to floating point
    rounding error in cancellations. This should only be done when it is known that all
    nonzero elements will be above this threshold. Operates on the arrays in-place and
    returns None"""
    threshold = 10 * np.finfo(float).eps * scale
    for arr in arrs:
        arr.real[np.abs(arr.real) < threshold] = 0
        if arr.dtype == np.complex:
            arr.imag[np.abs(arr.imag) < threshold] = 0


def _make_state_label(N, L, J):
    """Return a state label like 5P3/2 for a state with given principal, orbital angular
    momentum and total electronic angular momentum quantum numbers"""
    SPECTROSCOPIC_LABELS = {0: 'S', 1: 'P', 2: 'D', 3: 'F'}
    J = _halfint(J)
    if isinstance(J, int):
        Jstr = f'{J}'
    else:
        Jstr = f'{_int(2 * J)}/2'
    return f"{N}{SPECTROSCOPIC_LABELS[L]}{Jstr}"


def find_F(eigenval):
    return _halfint((-1 + np.sqrt(4 * eigenval / hbar ** 2 + 1)) / 2)


def find_mF(eigenval):
    return _halfint(eigenval / hbar)


def Hconj(A):
    """Hermitian conjugate of matrix in last two dimensions"""
    return A.conj().swapaxes(-1, -2)


def matrixel(u, A, v):
    """Matrix element between u and v, which are vectors in their last dimension, of A,
    which is a matrix in its last two dimensions"""
    return np.einsum('...i,...ij,...j', u.conj(), A, v)


def braket(u, v):
    """braket of u and v, which are vectors in their last dimension"""
    return np.einsum('...i,...i', u.conj(), v)


def ketbra(u, v):
    """ketbra of u and v, which are vectors in their last dimension"""
    return np.einsum('...i,...j', u.conj(), v)


def sorted_eigh(A):
    """np.linalg.eigh with results sorted by eigenvalue from smallest to largest"""
    evals, U = np.linalg.eigh(A)
    indices = evals.argsort(axis=-1)
    evals = np.take_along_axis(evals, indices, axis=-1)
    U = np.take_along_axis(U, indices[..., np.newaxis, :], axis=-1)
    return evals, U


def make_J_basis(J):
    """Construct a dict mapping mJ quantum numbers to basis vectors in the convention
    used by this module in which the |mJ> basis is ordered by mJ from highest to
    lowest."""
    mJ = _halfint(np.arange(J, -J - 1, -1))
    return {num: vec for num, vec in zip(mJ, np.identity(len(mJ)))}


def make_mJmI_basis(J, I):
    """Construct a dict mapping (mJ, mI) quantum numbers to basis vectors in the
    convention used by this module in which the |mJ, mI> basis is ordered first by mJ
    from highest to lowest; basis states with equal mJ are then ordered by mI from
    highest to lowest."""
    mJmI = outer(_halfint(np.arange(J, -J - 1, -1)), _halfint(np.arange(I, -I - 1, -1)))
    return {nums: vec for nums, vec in zip(mJmI, np.identity(len(mJmI)))}


def make_FmF_basis(J, I):
    """Construct a dict mapping (F, mF) quantum numbers corresponding to coupled spins J
    and I to basis vectors in the convention used by this module in which the |F, mF>
    basis is ordered first by F from highest to lowest; basis states with equal F are
    then ordered by mF from highest to lowest."""
    FmF = [
        (F, mF)
        for F in _halfint(np.arange(J + I, abs(J - I) - 1, -1))
        for mF in _halfint(np.arange(F, -F - 1, -1))
    ]
    return {nums: vec for nums, vec in zip(FmF, np.identity(len(FmF)))}


def ClebschGordan(j, m, j1, m1, j2, m2):
    """return the Clebsch-Gordan coeffienct <j, m|j1, m1; j2, m2>"""
    if m != m1 + m2:
        return 0
    # Workaround numpy issue https://github.com/numpy/numpy/issues/8917, Can't raise
    # integers to negative integer powers if the power is a numpy integer:
    j, m, j1, m1, j2, m2 = [float(x) for x in (j, m, j1, m1, j2, m2)]
    return (-1) ** (j1 - j2 + m) * np.sqrt(2 * j + 1) * Wigner3j(j1, j2, j, m1, m2, -m)


def U_CG(J, I):
    """Construct the unitary of Clebsch-Gordan coefficients that transforms a vector
    from the |mJ, mI> basis into the |F, mF> basis for a given J and I. The convention
    used for ordering the basis vectors is as described by make_mJmI_basis() and
    make_FmF_basis()."""
    return sum(
        ClebschGordan(F, mF, J, mJ, I, mI) * ketbra(FmFvec, mJmIvec)
        for ((F, mF), FmFvec), ((mJ, mI), mJmIvec) in outer(
            make_FmF_basis(J, I).items(), make_mJmI_basis(J, I).items()
        )
    )


def angular_momentum_operators(J):
    """Construct matrix representations of the angular momentum operators Ĵ = (Ĵx,
    Ĵy, Ĵz) and Ĵ² in the eigenbasis of Ĵz for a system with total angular momentum
    quantum number J. The basis is ordered first by m_I from highest to lowest; basis
    states with equal m_I are then ordered by m_J from highest to lowest."""
    n_mJ = _int(2 * J + 1)
    mJlist = _halfint(np.linspace(J, -J, n_mJ))
    Jp = np.diag(
        [hbar * np.sqrt(J * (J + 1) - mJ * (mJ + 1)) for mJ in mJlist if mJ < J], 1
    )
    Jm = np.diag(
        [hbar * np.sqrt(J * (J + 1) - mJ * (mJ - 1)) for mJ in mJlist if mJ > -J], -1
    )
    Jx = (Jp + Jm) / 2
    Jy = (Jp - Jm) / 2j
    Jz = np.diag([hbar * mJ for mJ in mJlist])
    J2 = Jx @ Jx + Jy @ Jy + Jz @ Jz
    _cleanarrays(hbar, Jx, Jy, Jz)
    _cleanarrays(hbar**2, J2)
    return Jx, Jy, Jz, J2


def angular_momentum_product_space(J, I):
    """Compute angular momentum operators for the product space of two systems with
    total angular momentum quantum numbers J and I. Return the vector angular momentum
    operators Ĵ, Î and F̂ = Ĵ + Î as well as Ĵ², Î² and F̂². All operators are
    returned in the simultaneous eigenbasis of F̂z and F̂². The basis is ordered first
    by F from highest to lowest; states with equal F are then ordered by mF by highest
    to lowest. Returns: (Jx, Jy, Jz, J2), (Ix, Iy, Iz, I2), (Fx, Fy, Fz, F2)"""

    # Operators in the individual subspaces:
    Jx, Jy, Jz, J2 = angular_momentum_operators(J)
    Ix, Iy, Iz, I2 = angular_momentum_operators(I)

    # The transformation into the |F, mF> basis (the basis that diagonalises F2 and Fz):
    U = U_CG(J, I)

    # Identity matrices for the subspaces
    II_J = np.identity(_int(2 * J + 1))
    II_I = np.identity(_int(2 * I + 1))

    # Promote operators into the J × I product space and transform into the |F, mF>
    # basis:
    Jx, Jy, Jz, J2 = [U @ np.kron(Ja, II_I) @ Hconj(U) for Ja in [Jx, Jy, Jz, J2]]
    Ix, Iy, Iz, I2 = [U @ np.kron(II_J, Ia) @ Hconj(U) for Ia in [Ix, Iy, Iz, I2]]

    # The total angular momentum operator:
    Fx = Jx + Ix
    Fy = Jy + Iy
    Fz = Jz + Iz
    F2 = Fx @ Fx + Fy @ Fy + Fz @ Fz

    _cleanarrays(hbar, Jx, Jy, Jz, Ix, Iy, Iz, Fx, Fy, Fz)
    _cleanarrays(hbar**2, J2, I2, F2)

    return (Jx, Jy, Jz, J2), (Ix, Iy, Iz, I2), (Fx, Fy, Fz, F2)


def reduced_dipole_moment_J(Jg, Je, omega_0, lifetime):
    """The reduced transition dipole moment <Jg||e r||Je> for an excited state to
    groundstate transition with electron angular momentum quantum numbers Je and Jg
    respectively, computed from the empirically known lifetime and angular frequency of
    the transition, with the convention that the reduced matrix element is real and
    positive. This phase convention is arbitrary so long as one does not consider
    transitions between multiple hyperfine excited states, in which case getting the
    relative phases of the different reduced dipole moments becomes important. For the D
    lines of the alkali metals, the phases of the reduced dipole moments for the two D
    lines are the same, so this convention is correct already. To extend this code to
    other lines where this might not be the case will require following the further
    decompositions of the dipole operator to figure out the relative phases."""
    return np.sqrt(
        ((2 * Je + 1) / (2 * Jg + 1))
        * ((3 * pi * epsilon_0 * hbar * c ** 3) / (lifetime * omega_0 ** 3))
    )


def dipole_operator(Jg, Je, I, omega_0, lifetime):
    """Compute the dipole operator with elements <Fe, mFe| e r_q| Fg, mFg> for
    transitions from a groundstate |Fg, mFg> to an excited state |Fe, mFe> via light
    with polarisation q = (-1, 0, 1). Returns a dictionary of three matrices, one for
    each q, each of dimension (2 Je + 1)(2 I + 1) × (2 Jg + 1)(2 I + 1). The dipole
    operator only couples states of different J, so you can think of each matrix as one
    of the off-diagonal blocks of a full matrix that has two blocks full of zeros on its
    diagonal (which would not be useful to construct). These matrices are not Hermitian,
    but since the dipole operator is a spherical tensor operator, elements of the other
    off-diagonal block (the matrix elements for transitions from excited states to
    groundstates) can be computed as:

    <Fg, mFg| e r_q| Fe, mFe> = (-1)^q <Fe, mFe| e r_{-q}| Fg, mFg>*.
    """

    # The reduced dipole moment for excited to ground transitions
    reduced_moment_J_e_to_g = reduced_dipole_moment_J(Jg, Je, omega_0, lifetime)
    # The reduced dipole moment for ground to excited transitions:
    reduced_moment_J = (
        (-1) ** (Jg - Je)
        * np.sqrt((2 * Jg + 1) / (2 * Je + 1))
        * reduced_moment_J_e_to_g.conj()
    )
    # Identity matrix of the nuclear subspace:
    II_I = np.identity(_int(2 * I + 1))
    # Unitaries for going from the |mJ, mI> basis to the |F, mF> basis in ground and
    # excited state manifolds:
    Ue = U_CG(Je, I)
    Ug = U_CG(Jg, I)
    dipole_operators = {}
    for q in [-1, 0, 1]:
        # First we compute the <Je, me| e r_q| Jg, mg> elements of the dipole operator.
        # Then we will transform to obtain <Fe, mFe| e r_q| Fg, mFg> elements.
        dipole_operator_J = sum(
            ClebschGordan(Je, me, Jg, mg, 1, q)
            * reduced_moment_J
            * ketbra(excited_vec, ground_vec)
            for (me, excited_vec), (mg, ground_vec) in outer(
                make_J_basis(Je).items(), make_J_basis(Jg).items()
            )
        )
        # Bump the operator up into the J × I product space and transform into the |F,
        # mF> basis of the ground and excited manifolds:
        dipole_operators[q] = Ue @ np.kron(dipole_operator_J, II_I) @ Hconj(Ug)
    return dipole_operators


class AtomicState(object):
    def __init__(
        self, I, J, gI, gJ, Ahfs, Bhfs=0, Bmax_crossings=500e-4, nB_crossings=5000
    ):
        self.I = I
        self.J = J
        self.Bmax_crossings = Bmax_crossings
        self.nB_crossings = nB_crossings
        J_ops, I_ops, F_ops = angular_momentum_product_space(J, I)
        self.Jx, self.Jy, self.Jz, self.J2 = J_ops
        self.Ix, self.Iy, self.Iz, self.I2 = I_ops
        self.Fx, self.Fy, self.Fz, self.F2 = F_ops
        self.basis_vectors = make_FmF_basis(J, I)

        # Identity matrix in the |F, mF> basis:
        II = np.identity(len(self.basis_vectors))

        # I dot J in units of hbar**2:
        rIJ = (self.Ix @ self.Jx + self.Iy @ self.Jy + self.Iz @ self.Jz) / hbar ** 2
        _cleanarrays(1, rIJ)
        self.H_hfs = Ahfs * rIJ
        if Bhfs != 0:
            # When Bhfs is zero, this term has a division by zero that doesn't get
            # caught. Subsequently multiplying by zero just turns the Infs into Nans. So
            # to avoid getting a H_hfs full of NaNs, we have to skip over this term when
            # it doesn't apply.
            numerator = 3 * rIJ @ rIJ + 3 / 2 * rIJ - I * (I + 1) * J * (J + 1) * II
            denominator = 2 * I * (2 * I - 1) * J * (2 * J - 1)
            self.H_hfs += Bhfs * numerator / denominator

        # The following assumes that the sign convention is followed in
        # which gI has a negative sign and gJ a positive one:
        self.mu_x = -(gI * self.Ix + gJ * self.Jx) * mu_B / hbar
        self.mu_y = -(gI * self.Iy + gJ * self.Jy) * mu_B / hbar
        self.mu_z = -(gI * self.Iz + gJ * self.Jz) * mu_B / hbar

        # Lists of F and mF quantum numbers at small field for the states, sorted by
        # energy from lowest to highest. These are used to identify the states that come
        # back from numerical diagonalisation.
        _, U = sorted_eigh(self.Htot(0))
        evecs = Hconj(U)
        E_Fz = matrixel(evecs, self.Fz, evecs).real
        E_F2 = matrixel(evecs, self.F2, evecs).real
        self._mF_by_energy = find_mF(E_Fz)
        self._F_by_energy = find_F(E_F2)

        # A key to store cached zeeman crossings so that we don't have to recompute them
        # all the time. They will be stored in the system temporary directory.
        cache_key = repr((I, J, gI, gJ, Ahfs, Bhfs, Bmax_crossings, nB_crossings))
        cache_file = os.path.join(gettempdir(), 'alkali_zeeman_crossings_cache')
        with shelve.open(cache_file) as cache:
            if cache_key not in cache:
                cache[cache_key] = self._find_crossings()
            self.crossings = cache[cache_key]

    def Htot(self, Bz):
        """Return total Hamiltonian for the given z magnetic field. if Bz is an array,
        the returned array will have the matrix dimensions as the last two dimensions,
        and B_z's dimensions as the initial dimensions."""
        Bz = np.array(Bz, dtype=float)
        # Degenerate eigenstates at zero field make it impossible to calculate
        # simultaneous eigenstates of both Htot and Fz. We'll lift that degeneracy just
        # a little bit. This only affects the accuracy of the energy eigenvalues in the
        # 12th decimal place -- far beyond the accuracy of any of these calculations.
        Bz[Bz < 1e-15] = 1e-15
        return self.H_hfs - self.mu_z * Bz[..., np.newaxis, np.newaxis]

    def _find_crossings(self):
        B_range = np.linspace(0, self.Bmax_crossings, self.nB_crossings)
        evals = [
            sorted(np.linalg.eigh(self.Htot(B_range[0]))[0]),
            sorted(np.linalg.eigh(self.Htot(B_range[1]))[0]),
        ]
        crossings = []
        for Bz in tqdm(B_range[2:]):
            predicted = 2 * np.array(evals[-1]) - np.array(evals[-2])
            new_evals = np.array(sorted(np.linalg.eigh(self.Htot(Bz))[0]))
            for _, (a, b) in crossings:
                new_evals[a], new_evals[b] = new_evals[b], new_evals[a]
            prediction_failures = []
            for i, val in enumerate(new_evals):
                bestmatch = min(abs(val - predicted))
                if not bestmatch == abs(val - predicted[i]):
                    prediction_failures.append(i)
            if len(prediction_failures) == 2:
                crossings.append(
                    (Bz - 0.5 * (B_range[1] - B_range[0]), prediction_failures)
                )
                a, b = prediction_failures
                new_evals[a], new_evals[b] = new_evals[b], new_evals[a]
            evals.append(new_evals)
        return crossings

    def _solve(self, Bz):
        evals, U = sorted_eigh(self.Htot(Bz))
        # Now to apply some swapping to account for crossings at lower fields than we're
        # at. The states will then be sorted by the energy eigenvalues that they
        # converge to at low but nonzero field.
        for field, (a, b) in self.crossings:
            s = Bz > field
            evals[s, a], evals[s, b] = evals[s, b], evals[s, a]
            U[s, :, a], U[s, :, b] = U[s, :, b], U[s, :, a]
        return evals, U

    def energy_eigenstates(self, Bz):
        evals, U = self._solve(Bz)
        evecs = Hconj(U)
        results = {}
        for i, (alpha, mF) in enumerate(zip(self._F_by_energy, self._mF_by_energy)):
            # Impose phase convention that each eigenvector's inner product with the
            # corresponding zero field eigenvector is real and positive:
            evec = evecs[..., i, :]
            proj = braket(self.basis_vectors[alpha, mF], evec)[..., np.newaxis]
            evec /= proj / np.abs(proj ** 2)
            results[alpha, mF] = (evals[..., i], evec)
        return results

    def transitions(self, Bz):
        states = self.energy_eigenstates(Bz)
        transitions = {}
        for (alpha, mF), (E, _) in states.items():
            for (alphaprime, mFprime), (Eprime, _) in states.items():
                if abs(mFprime - mF) <= 1 and (alpha, mF) != (alphaprime, mFprime):
                    omega = (Eprime - E) / hbar
                    transitions[(alpha, mF), (alphaprime, mFprime)] = np.abs(omega)
        return transitions

    def rf_transition_matrix_element(
        self, alpha, mF, alphaprime, mFprime, direction, Bz
    ):
        states = self.energy_eigenstates(Bz)
        _, initial_state = states[alpha, mF]
        _, final_state = states[alphaprime, mFprime]
        magnetic_moments = {'x': self.mu_x, 'y': self.mu_y, 'z': self.mu_z}
        mu = magnetic_moments[direction]
        return matrixel(final_state, mu, initial_state)


class AtomicLine(object):
    def __init__(self, groundstate, excited_state, omega_0, lifetime):
        self.groundstate = groundstate
        self.excited_state = excited_state
        self.Jg = groundstate.J
        self.Je = excited_state.J
        self.I = groundstate.I
        self.omega_0 = omega_0
        self.lifetime = lifetime
        self.linewidth = 1 / lifetime
        self.dipole_operator = dipole_operator(
            self.Jg, self.Je, self.I, omega_0, lifetime
        )

    def transitions(self, Bz):
        groundstates = self.groundstate.energy_eigenstates(Bz)
        excited_states = self.excited_state.energy_eigenstates(Bz)
        transitions = {}
        for (alpha, mF), (E, _) in groundstates.items():
            for (alphaprime, mFprime), (Eprime, _) in excited_states.items():
                if abs(mFprime - mF) <= 1:
                    omega = (Eprime - E) / hbar
                    transitions[(alpha, mF), (alphaprime, mFprime)] = omega
        return transitions

    def transition_dipole_moment(self, alpha, mF, alphaprime, mFprime, Bz):
        """The dipole transition matrix element <alpha', mF'|e r_q|alpha, mF> in Coulomb
        metres, where q = mF' - mF, for a transition from a groundstate |alpha, mF> to
        an excited state |alpha', mF'>. The matrix element for the corresponding excited
        to groundstate transition can be computed as:

           (-1)^q × <alpha', mF'|e r_q|alpha, mF>*
        
        but note that in the rotating wave approximation the Rabi frequency is computed
        only from the ground-to-excited dipole transition matrix elements."""
        try:
            dipole_operator = self.dipole_operator[mFprime - mF]
        except KeyError:
            raise ValueError("require mF' - mF ∊ {-1, 0, 1}")
        _, groundstate = self.groundstate.energy_eigenstates(Bz)[alpha, mF]
        _, excited_state = self.excited_state.energy_eigenstates(Bz)[
            alphaprime, mFprime
        ]
        return matrixel(excited_state, dipole_operator, groundstate)


class FineStructureLine(object):
    def __init__(self, line1, line2, N, L, Nprime, Lprime):
        """Wrapper around two AtomicLine objects, assumed to share a groundstate with
        principal quantum number N and orbital angular momentum quantum number L, and
        where the two excited states have the same Nprime, Lprime, differing only by
        their J quantum number.  The N and L quantum numbers are used solely to label
        the transitions returned by transitions() with an additional string NLJ, with L
        in spectroscopic notation and J written as a fraction, i.e. 5P3/2"""
        self.line1 = line1
        self.line2 = line2
        self.groundstate_label = _make_state_label(N, L, line1.groundstate.J)
        self.excited_state_1_label = _make_state_label(
            Nprime, Lprime, line1.excited_state.J
        )
        self.excited_state_2_label = _make_state_label(
            Nprime, Lprime, line2.excited_state.J
        )

    def transitions(self, Bz):
        transitions = {}
        for (initial, final), freq in self.line1.transitions(Bz).items():
            initial = (self.groundstate_label,) + initial
            final = (self.excited_state_1_label,) + final
            transitions[(initial, final)] = freq
        for (initial, final), freq in self.line2.transitions(Bz).items():
            initial = (self.groundstate_label,) + initial
            final = (self.excited_state_2_label,) + final
            transitions[(initial, final)] = freq
        return transitions

    def transition_dipole_moment(self, alpha, m, Je, alphaprime, mprime, Bz):
        """The dipole transition matrix element <Je, alpha', mF'|e r_q|Jg, alpha, mF> in
        Coulomb metres, where q = mF' - mF, for a transition from a groundstate |Jg,
        alpha, mF> to an excited state |Je, alpha', mF'>. The matrix element for the
        corresponding excited to groundstate transition can be computed as:

           (-1)^q × <alpha', mF'|e r_q| J, alpha, mF>*

        but note that in the rotating wave approximation the Rabi frequency is
        computed only from the ground-to-excited dipole transition matrix elements."""
        if Je == self.line1.Je:
            return self.line1.transition_dipole_moment(alpha, m, alphaprime, mprime, Bz)
        elif Je == self.line2.Je:
            return self.line2.transition_dipole_moment(alpha, m, alphaprime, mprime, Bz)
        else:
            msg = 'Given Je does not match any of this line\'s transitions.'
            raise ValueError(msg)
