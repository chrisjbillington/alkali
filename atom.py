import os
from tempfile import gettempdir
import numpy as np
from .wigner import Wigner3j, Wigner6j
import shelve

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


def make_key(obj):
    """ For an arbitrarily nested list, tuple, set, or dict, convert all numpy arrays to
    tuples of their data and metadata, convert all lists and dicts to tuples, and store
    every item alongside its type. This creates an object that can be used as a
    dictionary key to represent the original types and data of the nested objects that
    might otherwise not be able to be used as a dictionary key due to not being
    hashable."""
    if isinstance(obj, (list, tuple)):
        return tuple(make_key(item) for item in obj)
    elif isinstance(obj, set):
        return set(make_key(item) for item in obj)
    elif isinstance(obj, dict):
        return tuple((key, make_key(value)) for key, value in obj.items())
    elif isinstance(obj, np.ndarray):
        return obj.tobytes(), obj.dtype, obj.shape
    else:
        return type(obj), obj


def lru_cache(maxsize=128):
    """Decorator to cache up to `maxsize` most recent results of a function call. Custom
    implementation instead of using `functools.lru_cache()`, so that we can create
    dictionary keys for unhashable types like numpy arrays, which do not work with
    `functools.lru_cache()`."""

    def decorator(func):
        import functools, collections

        cache = collections.OrderedDict()

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            cache_key = make_key((args, kwargs))
            try:
                result = cache[cache_key]
            except KeyError:
                try:
                    result = cache[cache_key] = func(*args, **kwargs)
                except Exception as e:
                    # We don't want the KeyError in the exception:
                    raise e from None
            cache.move_to_end(cache_key)
            while len(cache) > maxsize:
                cache.popitem(last=False)
            return result

        return wrapped

    return decorator


def get_gF(F, I, J, gI, gJ):
    term1 = gJ * (F * (F + 1) - I * (I + 1) + J * (J + 1)) / (2 * F * (F + 1))
    term2 = gI * (F * (F + 1) + I * (I + 1) - J * (J + 1)) / (2 * F * (F + 1))
    return term1 + term2


def outer(list_a, list_b):
    outer_ab = []
    for a in list_a:
        for b in list_b:
            outer_ab.append((a, b))
    return outer_ab


def _int(x):
    """Round x to an integer, returning an int dtype"""
    x = np.round(np.array(x))
    return np.array(x, dtype=int)


def _halfint(x):
    """Round x to a half-integer, returning an int if the result is a whole number and a
    float otherwise"""
    x = np.array(x)
    twice_x = np.round(2 * x)
    # Halve and decide whether to return as float or int (array)
    if np.any(twice_x % 2):
        return twice_x / 2
    else:
        return np.array(twice_x // 2, dtype=int)


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


def ClebschGordan(F, mF, I, J, mI, mJ):
    """return the Clebsch-Gordan coeffienct <F, mF|I, J, mI, mJ>"""
    if mF != mI + mJ:
        return 0
    # Work around numpy issue https://github.com/numpy/numpy/issues/8917, Can't raise
    # integers to negative integer powers if the power is a numpy integer:
    F, mF, I, J, mI, mJ = [float(x) for x in (F, mF, I, J, mI, mJ)]
    return (-1) ** int(I - J + mF) * np.sqrt(2 * F + 1) * Wigner3j(I, J, F, mI, mJ, -mF)


def dipole_moment_zero_field(F, mF, Fprime, mFprime, J, Jprime, I, lifetime, omega_0):
    """ Calculate the transition dipole moment in SI units (Coulomb metres) for given
    initial and final F, mF states of a hydrogenic atom. Also required are the initial
    and final J's, the nuclear spin, and the lifetime and angular frequency of the
    transition. This calculation assumes the primed quantum numbers correspond to the
    excited state, so it is computing the dipole transition moment from ground to
    excited state."""
    reduced_dipole_J = (-1) ** (Jprime - J) * np.sqrt(
        3 * pi * epsilon_0 * hbar * c ** 3 / (lifetime * omega_0 ** 3)
    )
    reduced_dipole_F = (
        (-1) ** (F + Jprime + 1 + I)
        * np.sqrt((2 * F + 1) * (2 * Jprime + 1))
        * Wigner6j(Jprime, J, 1, F, Fprime, I)
        * reduced_dipole_J
    )
    return ClebschGordan(Fprime, mFprime, F, 1, mF, mFprime - mF) * reduced_dipole_F

def make_mImJ_basis(I, J):
    """Construct a dict mapping (mI, mJ) quantum numbers to basis vectors in the
    convention used by this module in which the |mI, mJ> basis is ordered first by mI
    from highest to lowest; basis states with equal mI are then ordered by mJ from
    highest to lowest."""
    mImJ = outer(_halfint(np.arange(I, -I - 1, -1)), _halfint(np.arange(J, -J - 1, -1)))
    return {nums: vec for nums, vec in zip(mImJ, np.identity(len(mImJ)))}


def make_FmF_basis(I, J):
    """Construct a dict mapping (F, mF) quantum numbers to basis vectors in the
    convention used by this module in which the |F, mF> basis is ordered first by F from
    highest to lowest; basis states with equal F are then ordered by mF from highest to
    lowest."""
    FmF = [
        (F, mF)
        for F in _halfint(np.arange(I + J, abs(I - J) - 1, -1))
        for mF in _halfint(np.arange(F, -F - 1, -1))
    ]
    return {nums: vec for nums, vec in zip(FmF, np.identity(len(FmF)))}


def U_CG(I, J):
    """Construct the unitary of Clebsch-Gordan coefficients that transforms a vector
    from the |m_I, m_J> basis into the |F, m_F> basis for a given I and J. The
    convention used for ordering the basis vectors is as described by make_mImJ_basis()
    and make_FmF_basis()."""
    mImJ = make_mImJ_basis(I, J)
    FmF = make_FmF_basis(I, J)
    U = np.zeros((len(FmF), len(FmF)))
    for i, (F, mF) in enumerate(FmF):
        for j, (mI, mJ) in enumerate(mImJ):
            U[i, j] = ClebschGordan(F, mF, I, J, mI, mJ)
    return U


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
    return Jx, Jy, Jz, J2


def angular_momentum_product_space(I, J):
    """Compute angular momentum operators for the product space of two systems with
    total angular momentum quantum numbers I and J. Return the vector angular momentum
    operators Î, Ĵ and F̂ = Î + Ĵ as well as Î², Ĵ² and F̂². All operators are
    returned in the simultaneous eigenbasis of F̂z and F̂². The basis is ordered first
    by F from highest to lowest; states with equal F are then ordered by mF by highest
    to lowest. Returns: (Ix, Iy, Iz, I2), (Jx, Jy, Jz, J2), (Fx, Fy, Fz, F2)"""

    # Operators in the individual subspaces:
    Ix, Iy, Iz, I2 = angular_momentum_operators(I)
    Jx, Jy, Jz, J2 = angular_momentum_operators(J)

    # The transformation into the F, mF basis (the basis that diagonalises F2 and Fz):
    U = U_CG(I, J)

    # Identity matrices for the subspaces
    II_I = np.identity(_int(2 * I + 1))
    II_J = np.identity(_int(2 * J + 1))

    # Promote operators into the product space and transform into the F, mF basis:
    Ix, Iy, Iz, I2 = [U @ np.kron(Ia, II_J) @ Hconj(U) for Ia in [Ix, Iy, Iz, I2]]
    Jx, Jy, Jz, J2 = [U @ np.kron(II_I, Ja) @ Hconj(U) for Ja in [Jx, Jy, Jz, J2]]

    # The total angular momentum operator:
    Fx = Ix + Jx
    Fy = Iy + Jy
    Fz = Iz + Jz
    F2 = Fx @ Fx + Fy @ Fy + Fz @ Fz

    return (Ix, Iy, Iz, I2), (Jx, Jy, Jz, J2), (Fx, Fy, Fz, F2)


class AtomicState(object):
    def __init__(
        self, I, J, gI, gJ, Ahfs, Bhfs=0, Bmax_crossings=500e-4, nB_crossings=5000
    ):
        self.I = I
        self.J = J
        self.Bmax_crossings = Bmax_crossings
        self.nB_crossings = nB_crossings
        I_ops, J_ops, F_ops = angular_momentum_product_space(I, J)
        self.Ix, self.Iy, self.Iz, self.I2 = I_ops
        self.Jx, self.Jy, self.Jz, self.J2 = J_ops
        self.Fx, self.Fy, self.Fz, self.F2 = F_ops
        self.basis_vectors = make_FmF_basis(I, J)

        # Identity matrix in the F, mF basis:
        II = np.identity(_int((2 * I + 1) * (2 * I + 1)))

        # I dot J in units of hbar**2:
        rIJ = (self.Ix @ self.Jx + self.Iy @ self.Jy + self.Iz @ self.Jz) / hbar ** 2
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
        # 13th decimal place -- far beyond the accuracy of any of these calculations.
        Bz[Bz < 1e-15] = 1e-15
        return self.H_hfs - self.mu_z * Bz[..., np.newaxis, np.newaxis]

    def _find_crossings(self):
        B_range = np.linspace(0, self.Bmax_crossings, self.nB_crossings)
        evals = [
            sorted(np.linalg.eigh(self.Htot(B_range[0]))[0]),
            sorted(np.linalg.eigh(self.Htot(B_range[1]))[0]),
        ]
        crossings = []
        from tqdm import tqdm

        for Bz in tqdm(B_range[2:]):
            predicted = 2 * np.array(evals[-1]) - np.array(evals[-2])
            new_evals = np.array(sorted(np.linalg.eigh(self.Htot(Bz))[0]))
            for field, (a, b) in crossings:
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
        for i, (F, mF) in enumerate(zip(self._F_by_energy, self._mF_by_energy)):
            results[F, mF] = (evals[..., i], evecs[..., i, :])
        return results

    @lru_cache()
    def transitions(self, Bz):
        states = self.energy_eigenstates(Bz)
        transitions = {}
        for (alpha, mF), E in states.items():
            for (alphaprime, mFprime), Eprime in states.items():
                if abs(mFprime - mF) <= 1 and (alpha, mF) != (alphaprime, mFprime):
                    omega = (Eprime - E) / hbar
                    transitions[(alpha, mF), (alphaprime, mFprime)] = np.abs(omega)
        return transitions

    @lru_cache()
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
        self.J = groundstate.J
        self.Jprime = excited_state.J
        self.I = groundstate.I
        self.omega_0 = omega_0
        self.lifetime = lifetime
        self.linewidth = 1 / lifetime
        self.dipole_operator = self._make_dipole_operator()

    def _make_dipole_operator(self):
        """Construct the dipole operator e r_q = Σ |F' mF'><F' mF'|e r_q|F mF><F mF|
        where the sum is over F, mF, F', and mF'. The matrix constructed is not square,
        it is num_excited_states × num_groundstates. We compute three matrices, for q =
        (-1, 0, +1)."""
        N_ground = len(self.groundstate.basis_vectors)
        N_excited = len(self.excited_state.basis_vectors)
        dipole_operator = {q: np.zeros((N_excited, N_ground)) for q in [-1, 0, 1]}
        statepairs = outer(
            self.groundstate.basis_vectors.items(),
            self.excited_state.basis_vectors.items(),
        )
        for q in [-1, 0, 1]:
            for ((F, mF), groundvec), ((Fprime, mFprime), excited_vec) in statepairs:
                if (mFprime - mF != q) or abs(Fprime - F) > 1:
                    continue
                el = dipole_moment_zero_field(
                    F,
                    mF,
                    Fprime,
                    mFprime,
                    self.J,
                    self.Jprime,
                    self.I,
                    self.lifetime,
                    self.omega_0,
                )
                dipole_operator[q] += el * ketbra(excited_vec, groundvec)
        return dipole_operator

    @lru_cache()
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

    @lru_cache()
    def transition_dipole_moment(self, alpha, mF, alphaprime, mFprime, Bz):
        try:
            dipole_operator = self.dipole_operator[mFprime - mF]
        except KeyError:
            raise ValueError("require mF' - mF ∊ {-1, 0, 1}")
        _, groundstate = self.groundstate.energy_eigenstates(Bz)[alpha, mF]
        _, excited_state = self.excited_state.energy_eigenstates(Bz)[alphaprime, mFprime]
        return matrixel(excited_state, dipole_operator, groundstate).real


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

    def transition_dipole_moment(self, J, alpha, m, Jprime, alphaprime, mprime, Bz):
        if Jprime - J == self.line1.Jprime - self.line1.J:
            return self.line1.transition_dipole_moment(alpha, m, alphaprime, mprime, Bz)
        elif Jprime - J == self.line2.Jprime - self.line2.J:
            return self.line2.transition_dipole_moment(alpha, m, alphaprime, mprime, Bz)
        else:
            raise ValueError(
                'Given J and Jprime do not match any of this line\'s transitions.'
            )
